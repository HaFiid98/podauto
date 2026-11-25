Provider setup and diagnostics

Run commands from the project root with the virtual environment active. Inspect
configuration without spending API quota:

```bash
python -m podauto.cli providers
```

`configured` means the key variable is populated and the adapter can construct
the request. A live request is still needed to verify authentication, model
availability, and account allowance. Configuration errors now produce a concise
message and exit code 2.

For a new installation, copy `config/providers.example.json` to
`config/providers.json`. The example is intentionally disabled; enable the
providers you want and confirm their model licences and service terms. Keep
credential values in `.env`, with plain `NAME=value` entries. The JSON `keys`
list contains only those names. Both runtime files are ignored by Git.

| Provider | Key variable | Request and response handling |
| --- | --- | --- |
| Cloudflare | `CLOUDFLARE_API_KEY` | Flux.1 Schnell accepts `prompt` and `steps`; fixed-ratio output is padded locally |
| Pollinations | `POLLINATIONS_API_KEY` | Authenticated `https://gen.pollinations.ai/image/{prompt}`, bearer header, requested dimensions |
| Fal | `FAL_AI_API_KEY` | `Authorization: Key ...`, custom `image_size`, one PNG, inline data URI |
| Gemini | `GEMINI_API_KEY` | `generateContent` requests text/image modalities and 4:5 output, padded locally to 5:6 |
| SiliconFlow | `SILICONFLOW_API_KEY` | Provider-specific `image_size` field, followed by download of the returned image URL |

The default SiliconFlow Flux.1 Schnell request uses the documented 768×1024
portrait size. Padding adds a matching ground without resizing or cropping art.
Image downloads do not receive the generation provider's authentication header.
Generated artwork still needs image QA, print conversion, and human review.

The operator-local rotation now tries the verified Cloudflare and Pollinations
providers first. Nscale's configured image model returned HTTP 410 on
2026-11-24, and the Hugging Face catalogue listed no Nscale text-to-image models.
That runtime route is disabled with a reason; the example remains inert.

The remaining account actions found during the live checks on 2026-11-24 are:

- SiliconFlow: the configured `.cn` endpoint rejects the current key. Replace
  `SILICONFLOW_API_KEY` in `.env` with a credential from the active account. The
  checked `.com` endpoint reports that it is deprecated.
- Fal: its API reports an exhausted balance and a locked account. Add credits
  at <https://fal.ai/dashboard/billing> before retrying.
- Gemini: the selected model has zero image-generation free-tier quota for this
  project. Enable the needed billing/quota in the Google project. The local
  historical quota-zero placeholders have been removed; the API still enforces
  the actual allowance.

Once the accounts are ready, generate a small batch:

```bash
python -m podauto.cli images --limit 1
python -m podauto.cli imageqa --stage generated
python -m podauto.cli printready --method raster
python -m podauto.cli imageqa --stage print_ready
python -m podauto.cli review
```

`--limit` counts records, each of which can contain several variations. The
usage ledger retains cooldowns and exhausted keys until their reset. If an
account is replenished after a quota failure, its recorded exhaustion may also
need to be cleared before an immediate retry. Existing shell exports take
precedence over `.env` values.

API references used for the adapters:

- Cloudflare: <https://developers.cloudflare.com/workers-ai/models/flux-1-schnell/>
- Fal: <https://fal.ai/models/fal-ai/flux/schnell/api?platform=http>
- Pollinations: <https://gen.pollinations.ai/openapi.json>
- SiliconFlow: <https://docs.siliconflow.com/en/api-reference/images/images-generations.md>
- Gemini request schema: <https://generativelanguage.googleapis.com/$discovery/rest?version=v1beta>
