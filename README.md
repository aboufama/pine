# engine-launch

Pushing to this branch runs `.github/workflows/launch-pod.yml`, which creates the Pine Engine pod on RunPod using the
`RUNPOD_API_KEY` repository secret and writes the outcome to `launch/last_launch.json`. The pod then runs the code on
branch `engine-code` and reports to branch `engine-results`.
