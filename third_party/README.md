# Third-party source policy

Floating source copies не хранятся в этом каталоге. Upstream dependencies
устанавливаются только по revisions из `configs/revisions.lock.yaml`.

Если M1 потребует source checkout LeRobot, bootstrap создаст его вне Git
worktree, проверит `git rev-parse HEAD` и запишет фактический SHA в environment
manifest.
