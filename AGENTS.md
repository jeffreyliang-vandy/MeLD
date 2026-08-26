# AGENTS.md — operating restrictions for this repository

## HARD RESTRICTIONS:

start a new branch for a new session

Development therefore runs as a one-way loop:

```
 local machine                      GitHub                  server (real data)
   edit + pytest on data/    --push-->  code  --pull-->   detached checkout @ sha
                                                                  |
                                                          runs in $RUN_ROOT (outside repo)
                                                                  |
   runs/<id>/summary.json  <--scp-- _export/  <--whitelist+scrub---+
```

Never pull to main without escalation

Never delete anything on the remote server

Never read any data on the remote server, except summarized results from test script.
