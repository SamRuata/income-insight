-- 003_run_architecture.sql
-- The template's `runs` table describes a single-hidden-layer net (hidden_dim).
-- The assignment requires an MLP with 2+ hidden layers and a controlled
-- comparison of >=3 configurations, so a run needs to record its full
-- architecture, not just one width. Apply after 002.

alter table runs add column if not exists config_name   text;
alter table runs add column if not exists hidden_sizes  text;      -- e.g. '128,64,32'
alter table runs add column if not exists activation    text;      -- 'relu' | 'gelu'
alter table runs add column if not exists dropout       double precision;
alter table runs add column if not exists weight_decay  double precision;
alter table runs add column if not exists n_train       integer;
alter table runs add column if not exists n_test        integer;
alter table runs add column if not exists best_epoch    integer;
alter table runs add column if not exists notes         text;

-- hidden_dim is NOT NULL in 001 but is meaningless for a multi-layer net.
-- Keep the column (tests and existing code reference it) but let it be null;
-- hidden_sizes is now the source of truth.
alter table runs alter column hidden_dim drop not null;

create index if not exists idx_runs_config_name on runs (config_name);