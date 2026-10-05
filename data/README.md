# Bundled experiment data

These files are copied unchanged from the user-supplied `QR_candicate-main(3).zip` so the paper-aligned scripts can run without a separate dataset download.

## Hosts (`data/hosts/classical/`)

- `airplane.bmp`
- `girl.bmp`
- `lenna.bmp`
- `manhattan.bmp`
- `pepper.bmp`
- `safari.bmp`

The paper-aligned method automatically evaluates the determinant-capacity gate. `manhattan.bmp` is expected to be reported as ineligible for the 4096-bit payload with nominal repetition `r=3`; it is intentionally kept in the bundle so the exclusion is reproducible rather than hidden.

## Watermarks (`data/watermarks/`)

- `watermark_1.png`
- `watermark_2.png`

Both are loaded and binarized to 64×64 by the evaluation code.
