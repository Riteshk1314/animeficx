# Agent run report: Demo.mp4

Model: `qwen3-vl:4b-instruct` (local, Ollama) · 42 model calls · 23 tool calls · 51,855 prompt + 2,204 output tokens · 78.7 s in the model · 308.9 s total

## Screens

| Shot | Kind | What the vision model saw | Tone |
|---|---|---|---|
| 0 | title_card | This is an episode title card for a new episode of the anime, featuring the episode's title in Japanese text. | childish |
| 2 | sign | a screen showing a character selection or identification interface with Japanese text for characters and sounds | childish |

## Readings corrected

| OCR read | Agent reading | Model's reason (first sentence) |
|---|---|---|
| カピパラ | カピバラ | OCR misread パ as バ (a common look-alike error), but the correct word is カピバラ (capybara). |

## Translations

| Japanese (final) | Dictionary gloss given to the model | English | Tracks |
|---|---|---|---|
| アップデートするゾ | アップデート (update) + する = 為る (to do; to carry out; to perform) | Update It! | #2 |
| キャラのこえ | キャラ ((fictional) character) + の ((particle)) + こえ = 声 (voice) | Character Voice | #81 |
| カピバラ | カピバラ (capybara (Hydrochoerus hydrochaeris); greater capybara) | Capybara | #82 |
| はなすはやさ | はなす = 話す (to talk; to speak; to converse) + はやさ = 速さ (speed; velocity; quickness) | Speech Speed | #83 |
| イヌ | イヌ = 犬 (dog (Canis (lupus) familiaris); canine) | Dog | #84, #86 |
| ネコ | ネコ = 猫 (cat (esp. the domestic cat, Felis catus); feline) | Cat | #87, #91 |
| サル | サル = 猿 (monkey (esp. the Japanese macaque, Macaca fuscata); ape; non-human primate) | Monkey | #88, #92, #99 |
| パンダ | パンダ (panda) | Panda | #95, #100, #104 |
| クマ | クマ = 熊 (bear (any mammal of family Ursidae)) | Bear | #101, #105, #111 |
| ウシ | ウシ = 牛 (cattle (Bos taurus); cow; bull) | Cow | #112 |

## Quality checks and repairs

- Round 1: 18/18 tracks passed.

## Flagged for a human

- Shot 2 (frame 291): `いろきめろ` near x=403-781, y=583-856

## Metrics

| Metric | Value |
|---|---|
| Japanese detections, input → output (49 sampled frames) | 115 → 0 (100.0% removed) |
| PSNR outside edited regions | 40.57 dB |
| Share of frame edited (frames with text) | 6.67% |
| Change in edited boxes while source is static (output / input) | 0.028 / 0.164 |
