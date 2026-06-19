# Publish this as a standalone repo (3 commands)

Run from the root of your cloned `daily-contributions`:

```bash
# 1. Copy scaffold to a new directory
cp -r cuda-bandwidth-roofline ~/cuda-bandwidth-roofline
cd ~/cuda-bandwidth-roofline

# 2. Initialise and commit
git init -b main
git add .
git commit -m "feat: CUDA memory bandwidth benchmark with roofline analysis"

# 3. Create public repo and push (requires gh CLI: https://cli.github.com)
gh repo create jyotisaini-3/cuda-bandwidth-roofline --public --source=. --push
```

Then open https://github.com/jyotisaini-3/cuda-bandwidth-roofline and:
- Replace the results table in README.md with your actual GPU's output
- Pin the repo on your profile: github.com/jyotisaini-3 → Customize pins
