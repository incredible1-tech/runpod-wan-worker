# runpod-wan-worker

Wan2.1-T2V-1.3B text-to-video serverless worker for RunPod. Replaces the
earlier `runpod-ltx2-worker` attempt, which turned out not to be a simple
diffusers model at all (LTX-2.3 needs ~80GB of separate checkpoints and a
custom monorepo install - see that repo's git history). Wan2.1-T2V-1.3B is
genuinely diffusers-native (`WanPipeline.from_pretrained(...)`), Apache 2.0
licensed, and small enough (1.3B params, ~8GB VRAM per the official docs)
to run on a 16GB tier instead of forcing an expensive one.

## Deploying

1. RunPod console -> Serverless -> New Endpoint -> Container Image
2. Image: `ghcr.io/incredible1-tech/runpod-wan-worker:latest`
3. GPU: 16GB tier
4. Min workers: 0 (scale-to-zero)
5. Execution timeout: 300s
6. **Attach a Network Volume** mounted at `/runpod-volume` - the model
   (~29GB, mostly the UMT5 text encoder shared across every Wan size) is
   downloaded there on the first cold start and reused by every worker
   after that. Without a volume attached, every cold worker re-downloads
   the full model from scratch, which is slow and wasteful.
7. Exclude any Blackwell-generation GPU (e.g. "PRO 6000 MIG") from
   "Enabled GPU types" under Advanced settings unless you've separately
   confirmed the base image's torch build supports it - see
   runpod-ltx2-worker's history for why this matters.

## Request format

```json
{
  "input": {
    "prompt": "...",
    "negative_prompt": "...",
    "width": 832,
    "height": 480,
    "num_frames": 81,
    "fps": 16,
    "guidance_scale": 5.0,
    "seed": null
  }
}
```

## Response format

```json
{
  "video": "data:video/mp4;base64,...",
  "duration_seconds": 5.06,
  "resolution": "832x480",
  "fps": 16,
  "seed": 12345,
  "generation_time_seconds": 42.1
}
```

On failure: `{"error": "...", "traceback": "..."}`.

## Notes

- The UMT5 text encoder runs on CPU, not GPU (matching Wan's own
  `--t5_cpu` CLI flag) - only the transformer and VAE are moved to CUDA.
  This is what gets VRAM use down to the ~8GB the 1.3B model is documented
  to need, instead of ~14GB+ if everything stayed on GPU.
- Default `negative_prompt` explicitly excludes anime/illustration/CGI
  terms on top of Wan's own documented quality-control terms - Wan has no
  reliable default toward photorealism in English-mode prompts and drifts
  toward flat illustrated output without this.
- Only does text-to-video. Wan's image-conditioned modes (I2V, FLF2V)
  require the much larger 14B checkpoints - not included in this worker.
