# Launching Runpod pods: from Claude or over SSH?

*Saved from a Claude Code answer, 2026-11-06. Update since then: the self-stop and the
start-command launch are now in [train/candidates/run_a100.sh](../train/candidates/run_a100.sh)
(commit `9718895`); its header has the exact start command.*

Do both, but for different jobs. Start and stop pods from Claude. For a batch run, have the
pod run the script itself rather than typing it over SSH.

## From Claude (the Runpod MCP tools or `runpodctl`)

Use this to create, stop and terminate pods, check stock and prices, read pod logs and
watch costs. These tools manage infrastructure only. They can't open a shell on the pod or
copy files to it.

## Over SSH

Use this for interactive work, such as debugging a failure, poking at data, or
benchmarking chunk sizes. Run long jobs inside `tmux` or `nohup`, so a dropped connection
doesn't kill them.

## Batch jobs: the start-command pattern

For a batch job like `run_a100.sh`, Claude creates the pod with the script as its start
command. That has three advantages:

- **It's reproducible.** The pod is defined entirely by its image, GPU, environment
  (`HF_TOKEN` as a Runpod secret) and start command, with nothing typed by hand.
- **It can be monitored without SSH.** Claude reads the pod's logs and reports the recall
  lines when they appear.
- **It can stop itself.** The script ends with `runpodctl stop pod $RUNPOD_POD_ID`. A pod
  keeps billing at the GPU rate after the script finishes until someone stops it, which is
  the easiest way to waste money.

SSH in only when something goes wrong.

## For a run started by hand

- If it was started in a plain SSH session, check it's inside `tmux`, `screen` or `nohup`.
  Otherwise disconnecting kills it, although a rerun resumes from what's already done.
- Stop the pod when it finishes, unless the script stops it itself.
