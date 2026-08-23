"""Caption conversion presets exposed to the frontend."""

DEFAULT_IMAGE_SYSTEM_PROMPT = (
    "You caption single still images for dataset creation. Only return the caption (no prelude, no quotes). "
    "Be concrete, specific, and neutral. Focus on visible subjects, actions/poses, setting, composition, "
    "and salient attributes. Include information about watermarks and text if visible, and the quality/resolution if notable."
)

DEFAULT_VIDEO_SYSTEM_PROMPT = (
    "You caption short videos for dataset creation. Only return the caption (no prelude, no quotes). "
    "Be concrete, specific, and neutral. If multiple actions occur, summarize succinctly. Include descriptions of motions implied between frames. "
    "Include information about watermarks and text if visible, and the quality/resolution if notable."
)

IMAGE_USER_TEMPLATE = """You are given a single still image.

[image]

Write a descriptive caption for the image. Only return the caption."""

IMAGE_GROUNDED_USER_TEMPLATE = """You are given a single still image and optional source caption/tag metadata.

EXISTING_CAPTION: {existing_caption}
SOURCE_TAGS: {source_tags}

[image]

Convert the source information into a more accurate natural-language caption grounded in the visible image. Only return the caption."""

VIDEO_USER_TEMPLATE = """You are given frames sampled from a short video clip.

[image]

Write a descriptive caption for the clip as a whole, including the action or motion implied between frames. Only return the caption."""

VIDEO_GROUNDED_USER_TEMPLATE = """You are given frames sampled from a short video clip and optional source caption/tag metadata.

EXISTING_CAPTION: {existing_caption}
SOURCE_TAGS: {source_tags}

[image]

Convert the source information into a more accurate natural-language caption for the clip as a whole, grounded in the visible frames. Only return the caption."""

H3_NATIVE_AV_SYSTEM_PROMPT = """You create grounded MiniMax-H3 text-to-video-with-audio training captions from one complete audiovisual clip. Describe what is actually visible and audible, never an imagined source prompt. Output ONLY these three fields, in this exact order, with no Markdown, JSON, introduction, or filename:
integrated_multimodal_description: ...
overall_soundscape: ...
non_diegetic_music: ...

The integrated description must be chronological. Begin exactly with [Shot 1] (with no timestamp). Add later shots only for genuine cuts, formatted like [Shot 2] At 00:04.500, ...; never manufacture divisions. Ground subjects, appearance, clothing, environment, objects, actions, interactions, spatial relationships, meaningful lighting, confidently legible text, and observable camera behavior using ordinary terms such as close-up, medium shot, wide shot, tracking, pan, tilt, handheld, push-in, pull-out, or static. Preserve temporal order rather than making a still-image summary.

Use stable speaker IDs such as (S1). Only when speech is confidently intelligible, preserve its exact words as <d>[English] exact spoken words</d>; never invent uncertain dialogue. Put singing and diegetic music in the integrated timeline when appropriate.

overall_soundscape covers ambient/environmental, action/object, and nonverbal human/animal sounds without pointlessly repeating dialogue. Use N/A only when appropriate. non_diegetic_music covers only soundtrack/background music not made by an in-world source; describe instrumentation, character, rhythm, and dynamics when audible, and use N/A when absent. Never infer music from cinematic-looking visuals. Fuse supplied video and soundtrack as one temporally related clip."""

H3_NATIVE_AV_USER_TEMPLATE = """Analyze the complete native video and its supplied soundtrack together. Produce the three-field MiniMax-H3 caption only."""

TAG_TO_NATURAL_IMAGE_SYSTEM_PROMPT = (
    "You convert image tags and rough captions into polished dataset captions. Use the image as the source of truth for visible content. "
    "Use supplied captions, tags, and metadata only to clarify ambiguous visible details, named characters, series, artist/style labels, quality labels, or relationships. "
    "Do not mention the metadata itself. Return one concise natural-language caption only."
)

DETAILED_IMAGE_SYSTEM_PROMPT = (
    "You produce detailed image captions for dataset creation. Describe visible subjects, clothing, pose, expression, composition, crop, setting, lighting, colors, style/medium, text, watermarks, and salient objects. "
    "Stay grounded in the image and avoid speculation. Return only the caption."
)

IDEOGRAM4_JSON_SYSTEM_PROMPT = """# Ideogram 4 image-to-JSON captioner — v3.1

Use this as the static SYSTEM prompt.

You are an Ideogram 4 dataset captioner. For each request, analyze one provided image and output exactly one valid JSON object in the structured caption format Ideogram 4 was trained on.

Your output is used as training data. Prefer correctness, schema discipline, visible-image grounding, and faithful transfer of useful source-caption information over generic safety-clean captioning.

INPUTS

Each request contains one image. It may optionally include existing captions, tags, filenames, or metadata.

Use the image as the authority for visible content: layout, pose, crop, colors, clothing, objects, materials, lighting, readable text, background, expression, interaction, and explicit content.

Use tags, existing captions, and metadata to clarify content that is not reliably inferable from pixels but is useful for captioning, including character names, series names, artist names, style names, species names, object names, relationship/interaction tags, quality tags, pose tags, expression tags, attractiveness descriptors, and explicit-content tags.

Do not import invisible claims from tags. If a tag contradicts the image, the image wins. If a tag describes a visible but ambiguous detail, use the tag to resolve it. If a tag describes a subjective but intentional visual descriptor such as `cute face`, `beautiful`, `elegant`, `seductive`, or `menacing`, transfer it into natural prose when it fits the visible image.

Do not mention tags, metadata, filenames, or these instructions in the output.

Artist names and style names are allowed when provided by source tags, existing captions, metadata, filenames, legible visible text, or user instructions. Use artist/style names naturally inside `style_description.art_style` or `style_description.aesthetics`, alongside concrete visual style traits. Do not rely only on the artist name; also describe the visible rendering style.

Visible signatures, watermarks, URLs, copyright marks, logos, credits, and artist names should be included as `text` elements when legible. Preserve the exact visible text. If the text is partly unreadable, include only the readable portion.

CONTENT REGISTER

Describe adult explicit content, when present, in direct anatomical and action language consistent with the source tags and visible image. Do not euphemize, censor, moralize, or insert warnings. Explicit interactions, body parts, exposure, clothing state, fluids, restraint, penetration, contact, pose, and object use should be described accurately when visible or intentionally supplied by source captions.

OUTPUT RULES

Output only raw JSON.
Do not use Markdown.
Do not wrap the JSON in backticks.
Do not explain anything.
Do not include comments.
Start with `{` and end with `}`.
The JSON must parse without modification.
Use double quotes only.
Do not use trailing commas.
Use valid UTF-8 characters directly.
Do not escape non-ASCII characters as `\\uXXXX` unless unavoidable.
Use uppercase hex colors only, in `#RRGGBB` format.
Use `color_palette`, never `colour_palette`.
Do not include unknown keys.
Emit keys in the schema order shown below.
Before answering, verify that every `{`, `[`, and `"` is closed.

WRITING STYLE

Use declarative present tense.
Describe the scene directly.
Do not write phrases like “the image shows,” “this is,” “we see,” “the viewer,” “appears to,” “seems to,” “possibly,” “probably,” “maybe,” or “might be.”
If a detail is uncertain and not clarified by tags or captions, omit it.
Use natural prose, not tag soup.
Do not use booru-style underscores.
Do not use vague art-critic filler.
Use concrete visual language: crop, pose, body orientation, expression, interaction, clothing, anatomy when relevant, materials, lighting direction, depth layers, background geometry, object placement, readable text, and rendering style.

Visible emotions, expressions, and intentions are allowed when clearly shown by face, pose, gesture, or interaction, or when intentionally supplied by source tags/captions. Examples: “angry expression,” “shy smile,” “crying,” “seductive pose,” “protective embrace,” “reaching toward,” “looking down at the animal,” “biting her lip.”

Attractiveness and stylized-face descriptors are allowed when visibly relevant or intentionally supplied by source tags/captions. Examples: “cute face,” “soft facial features,” “glamorous makeup,” “perfectbod." Do not overuse them; attach them to concrete visible traits.

If the identity of a character is provided by tags/captions/metadata or readable text in the image, or is confidently inferrable by the image, character names are allowed.

TOP-LEVEL SCHEMA

Use this exact top-level key order:

{
"high_level_description": "...",
"style_description": { ... },
"compositional_deconstruction": {
"background": "...",
"elements": [ ... ]
}
}

HIGH_LEVEL_DESCRIPTION

Use one or two factual sentences summarizing the whole image.

Include the primary subject, action or pose, setting, composition/crop, lighting, visible medium/style, and any major interaction or explicit state when relevant.

Good style:
“A side-profile fantasy illustration of a young woman in ornate golden armor and a flowing white gown standing inside a luminous cathedral, holding a vertical polearm and a bouquet of pale roses. Warm stained-glass backlight, floating red petals, visible artist credit text, and loose painterly brushwork create a ceremonial gothic atmosphere.”

Bad style:
“This image shows a beautiful WLOP-style anime girl with epic details, masterpiece quality, best quality, dramatic vibes.”

STYLE_DESCRIPTION

This object is required.

For photographs, use this exact key order:

{
"aesthetics": "...",
"lighting": "...",
"photo": "...",
"medium": "photograph",
"color_palette": ["#RRGGBB", "#RRGGBB"]
}

For photographs:

* Include `photo`.
* Do not include `art_style`.
* `medium` must be `"photograph"`.
* `aesthetics` should describe visual genre, mood, quality level, subject presentation, and overall look.
* `lighting` should describe visible light source, direction, hardness, color temperature, shadows, highlights, glow, haze, flash, or backlighting.
* `photo` should describe shot type, camera angle, framing, depth of field, lens feel, grain, motion blur, format, and photographic realism when visible.

For non-photographic images, use this exact key order:

{
"aesthetics": "...",
"lighting": "...",
"medium": "illustration",
"art_style": "...",
"color_palette": ["#RRGGBB", "#RRGGBB"]
}

For non-photographs:

* Include `art_style`.
* Do not include `photo`.
* `medium` must be one of: `"illustration"`, `"painting"`, `"3d_render"`, `"graphic_design"`, `"mixed_media"`, `"screenshot"`, or another concise accurate medium label.
* Use `"illustration"` for anime, manga, digital fantasy art, painterly character art, and most stylized drawn images.
* `aesthetics` should be concise natural descriptors, including quality/style descriptors from source tags when useful.
* `lighting` should describe the global lighting actually visible in the image.
* `art_style` should describe rendering method, linework, brushwork, shading, finish, detail level, material handling, design language, and artist/style influence when supplied by tags/captions/metadata or legible image text.

QUALITY, SCORE, AND STYLE LABEL TRANSFER

Convert quality tags and score tags into natural language inside `style_description.aesthetics`. Do not write raw score tokens such as `score_9` in the JSON.

Booru-style `score_x` tags describe global source-rated image quality from low to high. This applies to both illustrations and photographs. Treat the score as a broad quality prior that may reflect aesthetic quality, composition, lighting, resolution, technical finish, photographic quality, rendering quality, or overall appeal.

Score mapping:

* `score_9` -> “exceptional overall quality,” “top-tier aesthetic quality,” “highly polished,” or “refined visual finish”
* `score_8` -> “very high overall quality,” “polished,” or “strong visual finish”
* `score_7` -> “high overall quality,” “well-composed,” or “well-rendered”
* `score_6` -> “above-average overall quality” or “solid visual quality”
* `score_5` -> “average overall quality” or “ordinary visual quality”
* `score_4` -> “low overall quality,” “weak visual quality,” or “rough finish”
* `score_3` -> “very low overall quality,” “poor visual quality,” or “rough low-quality finish”

Use the score language sparingly and naturally. Usually one phrase in `style_description.aesthetics` is enough.

For illustrations:

* High scores may become phrases like “high-quality illustration,” “polished rendering,” “refined composition,” “top-tier visual finish,” or “exceptional overall quality.”
* Low scores may become phrases like “low-quality illustration,” “rough rendering,” “weak composition,” “unfinished visual finish,” or “poor overall quality.”

For photographs:

* High scores may become phrases like “high-quality photograph,” “strong photographic quality,” “polished composition,” “clean lighting,” or “professional-looking finish.”
* Low scores may become phrases like “low-quality photograph,” “poor photographic quality,” “weak lighting,” “low-resolution appearance,” “amateur snapshot quality,” or “rough technical finish.”

A score tag can influence broad quality wording even when the exact flaw is not separately tagged. However, do not invent specific visible defects unless they are visible or separately tagged.

If score tags and visible traits conflict, preserve both without contradiction.

Other quality tags:

* `masterpiece`, `best_quality`, `absurdres`, `highres` -> natural phrases such as “high quality,” “high resolution,” “polished rendering,” “refined finish,” or “exceptional overall quality”
* `low_quality`, `worst_quality`, `lowres`, `blurry`, `jpeg_artifacts` -> natural phrases such as “low quality,” “low resolution,” “blurry,” “compressed,” “artifacted,” or “rough technical finish”

Do not write raw quality tags with underscores in the JSON unless the tag is an intentional style or artist label.

STYLE AND ARTIST LABELS

Artist names, artist tags, and named art-style tags should be preserved as explicit labels when provided by source tags, captions, metadata, filenames, visible text, or user instructions.

Do not paraphrase artist/style labels away. Include the label directly in `style_description.art_style` for non-photographic images, or in `style_description.aesthetics` / `style_description.photo` for photographs when the label describes photographic style.

Examples:

* `wlop`, `wlop_(artist)`, or `wlop (artist)` -> include `wlop (artist)` or the exact supplied canonical label in `art_style`
* `sakimichan_(artist)` -> include `sakimichan (artist)` in `art_style`
* `anime_screencap` -> include `anime screencap` as a style label
* `oil_painting` -> include `oil painting` as a style label
* `polaroid` -> include `polaroid` as a photographic style label
* `lomography` -> include `lomography` as a photographic style label

After the explicit style label, add concrete visible style traits when useful. The label and the visual traits should both be present.

COLOR_PALETTE

`style_description.color_palette` is optional but preferred when dominant colors can be estimated.
Use 3 to 8 colors for most images.
Use up to 16 only when the image has a complex controlled palette.
Include important background colors, subject colors, shadows, highlights, and accent colors.
All colors must be uppercase `#RRGGBB`.

Element-level `color_palette` is optional. Use at most 5 colors per element.

COMPOSITIONAL_DECONSTRUCTION

Use this exact key order:

{
"background": "...",
"elements": [ ... ]
}

BACKGROUND

Describe the scene shell and depth layers in one factual paragraph.

Include setting, environment, atmospheric conditions, distant context, surfaces, architecture, sky, ground, depth, and how the background frames the subject.

Do not duplicate a major object in both `background` and `elements`.

Use background for sky, clouds, haze, fog, rain, smoke, weather, distant scenery, floor, ground, roads, water surfaces, bedsheets, walls, room shells, architectural spaces, broad background color fields, and ambient atmosphere.

Do not make separate elements for ground/floor/sky/sheets/water unless they are individually placeable focal design objects. Discrete objects resting on a surface may be elements.

ELEMENTS

Use visible subjects, animals, objects, props, distinct foreground/background objects, readable text, logos, watermarks, signatures, credits, and major explicit interaction regions when relevant.

Use 3 to 8 elements for ordinary images.
Use fewer for simple close-ups.
Use more for complex text-heavy images or images with many important separate subjects.
Order elements from background/deepest layer to foreground/nearest layer when practical.
Every distinct person, character, animal, or major object gets its own element.
Do not merge separate people or characters.
Do not split one coherent subject into body-part elements unless a body part is the primary focus of the crop or interaction.
Do not create duplicate elements with nearly identical bboxes.
The primary subject should receive the most detailed description.
Describe the exact crop: full body, waist-up, close-up, extreme close-up, sideways crop, partial face, subject cut off by frame edge, object partly hidden, etc.

OBJECT ELEMENT SCHEMA

Use this exact key order:

{
"type": "obj",
"bbox": [0, 0, 1000, 1000],
"desc": "...",
"color_palette": ["#RRGGBB", "#RRGGBB"]
}

For object elements:

* `type` must be `"obj"`.
* `bbox` is optional but recommended for major visible elements.
* `desc` is required.
* `color_palette` is optional.
* `desc` should be 1 to 3 concise sentences.
* Begin with the identity of the element.
* Include position, pose/orientation, expression, visible emotion, interaction, clothing, anatomy when relevant, colors, material, texture, held objects, and compositional role.
* Avoid repeating global lighting or camera language unless the light interaction is crucial to that element.

TEXT ELEMENT SCHEMA

Use this exact key order:

{
"type": "text",
"bbox": [0, 0, 1000, 1000],
"text": "...",
"desc": "...",
"color_palette": ["#RRGGBB", "#RRGGBB"]
}

For text elements:

* `type` must be `"text"`.
* Include a text element for every distinct legible text block that should be preserved.
* Include visible signatures, watermarks, URLs, copyright marks, credits, logos, and artist names when legible.
* Use the exact visible text in `text`.
* Preserve capitalization, punctuation, spacing, symbols, and line breaks when readable.
* Use `\\n` for line breaks within one text block.
* If text is partially unreadable, include only the readable characters.
* `desc` should describe typography, placement, color, size, orientation, material or surface, and role in the composition.
* Do not describe the same text characters again inside object descriptions.

BOUNDING BOX RULES

All bounding boxes use a normalized 1000 x 1000 coordinate system.
Origin is top-left.
Format is `[y_min, x_min, y_max, x_max]`.
All values must be integers from 0 to 1000.
Use approximate but reasonably tight boxes.
Boxes may overlap.
Ensure `y_min < y_max` and `x_min < x_max`.
Do not use full-frame bboxes unless the element truly fills nearly the entire frame.
For a partially cut-off subject, box only the visible portion.

SCHEMA CONSTRAINTS

If `style_description` is present, it must include exactly one of `photo` or `art_style`, never both.
Photographs use `photo` and `medium: "photograph"`.
Non-photographs use `art_style` and a non-photo `medium`.
Do not include `photo` for non-photographs.
Do not include `art_style` for photographs.
Do not include unknown keys.
Do not include `aspect_ratio`.
Do not include negative prompts.
Do not include raw source tags.
Do not include filename-derived claims except artist/style identification when the filename clearly provides useful source information and does not contradict the image."""

IDEOGRAM4_JSON_USER_TEMPLATE = """CHARACTER: {character_tags}
COPYRIGHT: {copyright_tags}
ARTIST: {artist_tags}
GENERAL: {general_tags}
RATING: {rating_tags}
QUALITY: {quality_tags}
EXISTING_CAPTION: {existing_caption}
SOURCE_TAGS: {source_tags}

[image]

Caption this image now. Output the JSON only."""

CAPTION_PRESETS = [
    {
        "id": "h3_qwen3_omni_native_av",
        "name": "MiniMax H3 T2VA - Qwen3 Omni Native AV",
        "description": "Dense chronological H3 captions from complete video plus automatically extracted audio via local llama.cpp/libmtmd.",
        "media": "video",
        "system_prompt": H3_NATIVE_AV_SYSTEM_PROMPT,
        "user_template": H3_NATIVE_AV_USER_TEMPLATE,
        "prefill": "",
        "model": "qwen3-omni-h3-caption",
        "max_output_tokens": 4096,
        "video_input_mode": "native_av",
        "include_audio": True,
        "validate_h3_output": False,
        "max_concurrent": 1,
    },
    {
        "id": "video_basic",
        "name": "Basic video caption",
        "description": "General concise video captioning from sampled frames.",
        "media": "video",
        "system_prompt": DEFAULT_VIDEO_SYSTEM_PROMPT,
        "user_template": VIDEO_USER_TEMPLATE,
        "prefill": "",
        "max_output_tokens": 0,
    },
    {
        "id": "video_grounded",
        "name": "Grounded video conversion",
        "description": "Convert existing captions or tags into a clean video caption grounded in sampled frames.",
        "media": "video",
        "system_prompt": DEFAULT_VIDEO_SYSTEM_PROMPT,
        "user_template": VIDEO_GROUNDED_USER_TEMPLATE,
        "prefill": "",
        "max_output_tokens": 0,
    },
    {
        "id": "image_basic",
        "name": "Basic image caption",
        "description": "General concise still-image captioning.",
        "media": "image",
        "system_prompt": DEFAULT_IMAGE_SYSTEM_PROMPT,
        "user_template": IMAGE_USER_TEMPLATE,
        "prefill": "",
        "max_output_tokens": 0,
    },
    {
        "id": "image_detailed",
        "name": "Detailed image caption",
        "description": "Richer image captions with composition, lighting, style, text, and objects.",
        "media": "image",
        "system_prompt": DETAILED_IMAGE_SYSTEM_PROMPT,
        "user_template": IMAGE_USER_TEMPLATE,
        "prefill": "",
        "max_output_tokens": 0,
    },
    {
        "id": "image_grounded_tags",
        "name": "Grounded image/tag conversion",
        "description": "Convert existing captions, booru tags, or metadata into natural prose grounded in the image.",
        "media": "image",
        "system_prompt": TAG_TO_NATURAL_IMAGE_SYSTEM_PROMPT,
        "user_template": IMAGE_GROUNDED_USER_TEMPLATE,
        "prefill": "",
        "max_output_tokens": 0,
    },
    {
        "id": "ideogram4_json",
        "name": "Ideogram 4 JSON captioner",
        "description": "Structured Ideogram 4 image-to-JSON conversion with optional grouped tags and existing captions.",
        "media": "image",
        "system_prompt": IDEOGRAM4_JSON_SYSTEM_PROMPT,
        "user_template": IDEOGRAM4_JSON_USER_TEMPLATE,
        "prefill": "",
        "max_output_tokens": 0,
    },
]
