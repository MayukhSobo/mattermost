// Copyright (c) 2015-present Mattermost, Inc. All Rights Reserved.
// See LICENSE.txt for license information.

package imaging

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"sync"

	_ "golang.org/x/image/bmp"
	_ "golang.org/x/image/tiff"
	_ "golang.org/x/image/webp"
)

const (
	riffChunkHeaderSize = 8  // FourCC (4 bytes) + uint32 data size (4 bytes)
	riffContainerSize   = 12 // "RIFF" + uint32 total size + "WEBP" FourCC
	anmfFrameHeaderSize = 16 // Frame X, Y, Width, Height, Duration, Flags
	vp8xPayloadSize     = 10 // Flags (4 bytes) + Canvas W-1 (3 bytes) + Canvas H-1 (3 bytes)
)

// DecoderOptions holds configuration options for an image decoder.
type DecoderOptions struct {
	// The level of concurrency for the decoder. This defines a limit on the
	// number of concurrently running encoding goroutines.
	ConcurrencyLevel int
}

func (o *DecoderOptions) validate() error {
	if o.ConcurrencyLevel < 0 {
		return errors.New("ConcurrencyLevel must be non-negative")
	}
	return nil
}

// Decoder holds the necessary state to decode images.
// This is safe to be used from multiple goroutines.
type Decoder struct {
	sem  chan struct{}
	opts DecoderOptions
}

// NewDecoder creates and returns a new image decoder with the given options.
func NewDecoder(opts DecoderOptions) (*Decoder, error) {
	var d Decoder
	if err := opts.validate(); err != nil {
		return nil, fmt.Errorf("imaging: error validating decoder options: %w", err)
	}
	if opts.ConcurrencyLevel > 0 {
		d.sem = make(chan struct{}, opts.ConcurrencyLevel)
	}
	d.opts = opts
	return &d, nil
}

// Decode decodes the given encoded data and returns the decoded image.
func (d *Decoder) Decode(rd io.Reader) (img image.Image, format string, err error) {
	if d.opts.ConcurrencyLevel != 0 {
		d.sem <- struct{}{}
		defer func() { <-d.sem }()
	}

	img, format, err = image.Decode(rd)
	if err != nil {
		return nil, "", fmt.Errorf("imaging: failed to decode image: %w", err)
	}

	return img, format, nil
}

// DecodeMemBounded works similarly to Decode but also returns a release function that
// must be called when access to the raw image is not needed anymore.
// This sets the raw image data pointer to nil in an attempt to help the GC to re-use the underlying data as soon as possible.
func (d *Decoder) DecodeMemBounded(rd io.Reader) (img image.Image, format string, releaseFunc func(), err error) {
	if d.opts.ConcurrencyLevel != 0 {
		d.sem <- struct{}{}
		defer func() {
			if err != nil {
				<-d.sem
			}
		}()
	}

	img, format, err = image.Decode(rd)
	if err != nil {
		return nil, "", nil, fmt.Errorf("imaging: failed to decode image: %w", err)
	}

	var once sync.Once
	releaseFunc = func() {
		if d.opts.ConcurrencyLevel == 0 {
			return
		}
		once.Do(func() {
			if img != nil {
				releaseImageData(img)
			}
			<-d.sem
		})
	}

	return img, format, releaseFunc, nil
}

// DecodeConfig returns the image config for the given data.
func (d *Decoder) DecodeConfig(rd io.Reader) (image.Config, string, error) {
	img, format, err := image.DecodeConfig(rd)
	if err != nil {
		return image.Config{}, "", fmt.Errorf("imaging: failed to decode image config: %w", err)
	}
	return img, format, nil
}

// GetDimensions returns the dimensions for the given encoded image data.
func GetDimensions(imageData io.Reader) (width int, height int, err error) {
	cfg, _, err := image.DecodeConfig(imageData)
	width, height = cfg.Width, cfg.Height
	if seeker, ok := imageData.(io.Seeker); ok {
		_, err2 := seeker.Seek(0, 0)
		if err == nil && err2 != nil {
			err = fmt.Errorf("failed to seek back to the beginning of the image data: %w", err2)
		}
	}
	return
}

// DecodeWebPFirstFrame decodes the first frame of an animated WebP.
// It streams the RIFF chunk list so only the first ANMF frame payload is read
// into memory — the rest of the file is discarded without buffering.
func (d *Decoder) DecodeWebPFirstFrame(r io.Reader) (image.Image, error) {
	if d.opts.ConcurrencyLevel != 0 {
		d.sem <- struct{}{}
		defer func() { <-d.sem }()
	}

	var header [riffContainerSize]byte
	if _, err := io.ReadFull(r, header[:]); err != nil {
		return nil, fmt.Errorf("webp: read failed: %w", err)
	}
	if string(header[:4]) != "RIFF" || string(header[riffChunkHeaderSize:]) != "WEBP" {
		return nil, errors.New("webp: not a WebP file")
	}

	var chunkHdr [riffChunkHeaderSize]byte
	for {
		if _, err := io.ReadFull(r, chunkHdr[:]); err != nil {
			break
		}
		id := string(chunkHdr[:4])
		size := int(binary.LittleEndian.Uint32(chunkHdr[4:]))

		if id == "ANMF" && size >= anmfFrameHeaderSize+riffChunkHeaderSize {
			payload := make([]byte, size)
			if _, err := io.ReadFull(r, payload); err != nil {
				break
			}
			container, err := anmfContainer(payload)
			if err != nil {
				return nil, err
			}
			img, _, err := image.Decode(bytes.NewReader(container))
			if err != nil {
				return nil, fmt.Errorf("webp: first frame decode failed: %w", err)
			}
			return img, nil
		}

		skip := int64(size)
		if size%2 != 0 {
			skip++
		}
		if _, err := io.CopyN(io.Discard, r, skip); err != nil {
			break
		}
	}
	return nil, errors.New("webp: no decodable animation frame found")
}

// anmfContainer builds a standalone RIFF/WEBP container from an ANMF chunk payload.
// The payload begins with the 16-byte ANMF frame header followed by subchunks.
// For lossy VP8 frames that carry an ALPH subchunk, it builds the extended format
// (VP8X + ALPH + VP8) so the golang webp decoder preserves transparency.
func anmfContainer(anmf []byte) ([]byte, error) {
	if len(anmf) < anmfFrameHeaderSize+riffChunkHeaderSize {
		return nil, errors.New("webp: ANMF payload too short")
	}
	// ANMF header bytes 6-8: frame width minus one; bytes 9-11: frame height minus one.
	wMinus1 := uint32(anmf[6]) | uint32(anmf[7])<<8 | uint32(anmf[8])<<16
	hMinus1 := uint32(anmf[9]) | uint32(anmf[10])<<8 | uint32(anmf[11])<<16

	sub := anmf[anmfFrameHeaderSize:]
	var alph []byte
	for pos := 0; pos+riffChunkHeaderSize <= len(sub); {
		id := string(sub[pos : pos+4])
		size := int(binary.LittleEndian.Uint32(sub[pos+4 : pos+riffChunkHeaderSize]))
		end := pos + riffChunkHeaderSize + size
		if end > len(sub) {
			break
		}
		switch id {
		case "ALPH":
			alph = sub[pos:end]
		case "VP8L":
			return webpContainer(sub[pos:end]), nil
		case "VP8 ":
			if alph != nil {
				return webpContainer(vp8xChunk(wMinus1, hMinus1), alph, sub[pos:end]), nil
			}
			return webpContainer(sub[pos:end]), nil
		}
		pos = end
		if size%2 != 0 {
			pos++
		}
	}
	return nil, errors.New("webp: no VP8/VP8L subchunk in ANMF frame")
}

// webpContainer wraps one or more chunks in a minimal RIFF/WEBP container.
func webpContainer(chunks ...[]byte) []byte {
	total := 4 // "WEBP"
	for _, c := range chunks {
		total += len(c)
	}
	buf := make([]byte, riffChunkHeaderSize+total)
	copy(buf, "RIFF")
	binary.LittleEndian.PutUint32(buf[4:], uint32(total))
	pos := riffChunkHeaderSize + copy(buf[riffChunkHeaderSize:], "WEBP")
	for _, c := range chunks {
		pos += copy(buf[pos:], c)
	}
	return buf
}

// vp8xChunk builds an 18-byte VP8X chunk for the extended WebP format.
// It sets only the alpha flag; all other feature bits are left zero.
func vp8xChunk(wMinus1, hMinus1 uint32) []byte {
	c := make([]byte, riffChunkHeaderSize+vp8xPayloadSize)
	copy(c, "VP8X")
	binary.LittleEndian.PutUint32(c[4:], vp8xPayloadSize)
	c[8] = 0x10 // alpha flag (bit 4 per the WebP spec)
	c[12] = byte(wMinus1)
	c[13] = byte(wMinus1 >> 8)
	c[14] = byte(wMinus1 >> 16)
	c[15] = byte(hMinus1)
	c[16] = byte(hMinus1 >> 8)
	c[17] = byte(hMinus1 >> 16)
	return c
}

// This is only needed to try and simplify GC work.
func releaseImageData(img image.Image) {
	switch raw := img.(type) {
	case *image.Alpha:
		raw.Pix = nil
	case *image.Alpha16:
		raw.Pix = nil
	case *image.Gray:
		raw.Pix = nil
	case *image.Gray16:
		raw.Pix = nil
	case *image.NRGBA:
		raw.Pix = nil
	case *image.NRGBA64:
		raw.Pix = nil
	case *image.Paletted:
		raw.Pix = nil
	case *image.RGBA:
		raw.Pix = nil
	case *image.RGBA64:
		raw.Pix = nil
	default:
		return
	}
}
