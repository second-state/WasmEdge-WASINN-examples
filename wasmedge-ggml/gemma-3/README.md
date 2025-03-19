# Gemma-3 Example For WASI-NN with GGML Backend

> [!NOTE]
> Please refer to the [wasmedge-ggml/README.md](../README.md) for the general introduction and the setup of the WASI-NN plugin with GGML backend. This document will focus on the specific example of the Gemma-3 model.

## Get Gemma-3 Model

```bash
curl -LO https://huggingface.co/second-state/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q5_K_M.gguf
curl -LO https://huggingface.co/second-state/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-mmproj-f16.gguf
```

## Prepare the Image

```bash
curl -LO https://llava-vl.github.io/static/images/monalisa.jpg
```

## Execute (with image file)

> [!NOTE]
> You may see some warnings stating `key clip.vision.* not found in file.`. These are expected and can be ignored.

```console
$ wasmedge --dir .:. \
  --env mmproj=gemma-3-4b-it-mmproj-f16.gguf \
  --env image=monalisa.jpg \
  --nn-preload default:GGML:AUTO:gemma-3-4b-it-Q5_K_M.gguf \
  wasmedge-ggml-gemme-3.wasm default
```

## Execute (with base64 encoded image)

> [!NOTE]
> You may see some warnings stating `key clip.vision.* not found in file.`. These are expected and can be ignored.

```console
$ wasmedge --dir .:. \
  --env mmproj=gemma-3-4b-it-mmproj-f16.gguf \
  --nn-preload default:GGML:AUTO:gemma-3-4b-it-Q5_K_M.gguf \
  wasmedge-ggml-gemme-3-base64.wasm default
```

## Results

```
USER:
Describe this image

ASSISTANT:
Okay, let's describe the image.

The image is a portrait of a man, almost certainly **Leonardo da Vinci's *Mona Lisa*.**

Here's a breakdown of the key features and overall impression:

*   **Subject:** A woman, believed to be Lisa del Giocondo, is the central figure. She is seated, turned slightly to the viewer, with her hands folded in her
lap.

*   **Composition:** She's positioned in a pyramidal form, creating a sense of stability and balance.

*   **Expression:** Her most famous feature is her enigmatic smile. It's subtle, almost ambiguous, and seems to shift depending on the angle of observation.
This is a key part of the painting's enduring mystery.

*   **Technique:** Da Vinci employed his signature *sfumato* technique - a subtle blending of colors and tones that creates a soft, hazy effect, particularly
around her eyes and mouth. This contributes to the dreamlike quality of the painting.

*   **Background:** A landscape is visible in the background, seemingly a hazy, distant vista of mountains and water. The landscape is atmospheric and slightl
y blurred, further drawing attention to the figure.

*   **Color Palette:** The painting employs a muted, earthy color palette – browns, greens, golds, and blues - giving it a timeless and serene quality.

*   **Overall Impression:** The *Mona Lisa* is a masterpiece of Renaissance art. It's renowned for its realism, psychological depth, and technical brilliance.
 It exudes an aura of mystery and beauty, which is why it's so widely recognized and studied.

**Do you want me to delve deeper into a specific aspect of the image, such as:**

*   The historical context of the painting?
*   The techniques Da Vinci used?
*   The theories surrounding her smile?

USER:
The techniques Da Vinci used?

ASSISTANT:
Okay, let’s dive deeper into the techniques Leonardo da Vinci employed to create the *Mona Lisa*. He was a meticulous and innovative artist, and the painting showcases several groundbreaking techniques he developed and perfected. Here’s a breakdown of the key ones:

**1. Sfumato:**

*   **What it is:** This is arguably the *most* famous technique associated with the *Mona Lisa*. “Sfumato” is an Italian word meaning “smoked” or “blurred.” It’s a subtle, almost imperceptible blending of colors and tones that creates a soft, hazy effect, particularly around the edges of forms and features.
*   **How he used it:** Da Vinci achieved this by applying incredibly thin layers of oil paint – often just a glaze – and meticulously blending them without harsh lines. He’d work with a tiny brush, gradually building up the tones to create a sense of depth and softness. You see it most prominently around her eyes and mouth, contributing to the elusive quality of her smile.

**2. Chiaroscuro:**

*   **What it is:** Chiaroscuro (Italian for "light-dark") is the use of strong contrasts between light and dark to create dramatic effects.
*   **How he used it:** Da Vinci uses chiaroscuro to model Mona Lisa’s face and body, giving her a three-dimensional appearance. The subtle gradations of light and shadow define her features and create a sense of volume.

**3. Layering and Glazing:**

*   **What it is:**  This technique involves applying many thin, translucent layers of paint (glazes) over a dry underpainting.
*   **How he used it:** Da Vinci built up the colors of the *Mona Lisa* through numerous thin glazes. Each layer subtly alters the color and tone of the layer beneath it. This created a luminous quality and depth of color that was far more vibrant than previous painting techniques. It also helped to create the *sfumato* effect.

**4. Aerial Perspective (Atmospheric Perspective):**

*   **What it is:** This technique uses variations in color and detail to create the illusion of depth in a landscape. Objects further away appear paler, less detailed, and bluer.
*   **How he used it:** The background landscape in the *Mona Lisa* demonstrates aerial perspective beautifully. The mountains and the distant water are rendered with muted colors and softened details, suggesting their
```