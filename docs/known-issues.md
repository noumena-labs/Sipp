# Known Issues

This page tracks current issues that users may hit when running Sipp.

## Browser Pulse Animations Can Reduce WebGPU Decode Throughput

Status: open.

Continuous pulse animations in the page can slow down browser-local inference,
especially WebGPU decode throughput. This has been observed as lower
tokens-per-second while a demo or app is rendering pulsing UI or scene effects
during generation.

Affected surface:

- Browser-local inference through the `sipp` browser package.
- Demos or applications that keep pulse animations active while the model is
  decoding.

Workarounds:

- Disable or pause pulse animations while a request is decoding.
- Prefer static state indicators or lower-frequency updates during generation.
- Test browser inference performance with visual animations disabled before
  comparing backend or model throughput.

## Hybrid Graphics Laptops May Pick The Integrated GPU

Status: open.

On Windows laptops with both integrated and discrete graphics, the browser may
choose the integrated GPU for WebGPU. Browser-local inference still runs, but
decode throughput can be much lower than expected.

Workaround:

1. Open **Windows Settings**.
2. Go to **System > Display > Graphics**.
3. Add the browser executable you use for Sipp, such as Chrome, Edge, or
   another Chromium-based browser.
4. Set that browser to **High performance**.
5. Restart the browser and reload the Sipp page.

This setting is stronger than relying on browser flags because it tells Windows
which GPU the browser process should prefer.
