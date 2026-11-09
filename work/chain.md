# The retrain chain, unattended

`/tmp/chain.sh` on `ds41-test-y` waits for the running dump and then runs
`STAGES="validate cache train export eval"` by itself. Log to `/data/mtp/chain.log` and
`/data/mtp/batch.out`.
