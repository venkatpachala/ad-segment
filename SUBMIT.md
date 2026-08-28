# Remaining human steps before the email

Code, labels, DESIGN, EVAL, README, tests, and viewer are in the repo. These you still have to do yourself:

## 1. Secrets (do this first)

`.env` was tracked locally. It is now gitignored. **Rotate** `OPENROUTER_API_KEY` and `OPENAI_API_KEY` — they lived in a working tree that should never be pushed.

Do not `git add .env`. Do not push the three old local commits (`b4872f4` … `b79afa5`); they contain multi-GB `data/cache` and `data/runs`.

## 2. Make the GitHub repo private and add the reviewer

The existing remote `https://github.com/venkatpachala/ad-segment` is **public**. The brief requires a **private** repo.

GitHub → the repo → Settings → Change repository visibility → Private.

Then Settings → Collaborators → Add **growth-droid** (pull is enough).

If GitHub CLI is installed:

```bash
gh repo edit venkatpachala/ad-segment --visibility private --accept-visibility-change-consequences
gh api -X PUT repos/venkatpachala/ad-segment/collaborators/growth-droid -f permission=pull
```

## 3. Push a clean commit, not the bloated history

After this machine’s reset-to-`origin/main` + one new commit:

```bash
git push origin main
```

If `git push` tries to upload 100MB+ mp4s, stop and check `git ls-files "*.mp4"`. Only `data/fixtures/*.mp4` should be listed.

## 4. Record the 5-minute walkthrough

Follow `RECORDING.md`. Upload unlisted (Drive / YouTube). Put the link in the email thread with the repo URL.

## 5. Reply on the thread

Repo: `https://github.com/venkatpachala/ad-segment` (private, growth-droid invited)

Recording: \<your link\>

One line on live: stream `s0LLVQeMmtU` was offline; 662s recording of the same channel, live path, documented in EVAL.md.
