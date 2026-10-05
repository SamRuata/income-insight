-- 002_adult_income.sql
-- Adds the real UCI Adult Income table and the columns the fairness audit needs.
-- Apply AFTER 001_init.sql in the Supabase dashboard: SQL Editor -> New query -> paste -> Run.

-- ---------------------------------------------------------------------------
-- adult_income: one row per training example from the real UCI Adult dataset.
-- Unlike the template's datasets.records JSONB blob, this is a real table, so
-- the fairness audit can JOIN and GROUP BY protected attributes in SQL.
-- ---------------------------------------------------------------------------
create table if not exists adult_income (
    id              bigint generated always as identity primary key,

    -- numeric features
    age             integer not null,
    education_num   integer not null,
    capital_gain    integer not null,
    capital_loss    integer not null,
    hours_per_week  integer not null,

    -- categorical features ("?" in the raw data is loaded as NULL and imputed
    -- by the sklearn pipeline, never silently dropped)
    workclass       text,
    marital_status  text,
    occupation      text,
    relationship    text,
    race            text,
    sex             text,
    native_country  text,

    -- target: raw string from the file, plus the 0/1 encoding the model uses
    income          text    not null,          -- '<=50K' or '>50K'
    label           integer not null,          -- 0 or 1
    split           text    not null default 'train',  -- 'train' or 'test'

    created_at      timestamptz not null default now()
);

create index if not exists idx_adult_income_sex    on adult_income (sex);
create index if not exists idx_adult_income_race   on adult_income (race);
create index if not exists idx_adult_income_split  on adult_income (split);
create index if not exists idx_adult_income_label  on adult_income (label);

-- ---------------------------------------------------------------------------
-- predictions: extra columns so a prediction can be scored for fairness later.
--
-- The template logs features/proba/label only, which is enough to report a
-- positive-prediction rate but NOT a false-positive or false-negative rate --
-- those need the ground truth. source_row_id links a prediction back to the
-- row it was made from; true_label caches that row's label so the audit query
-- stays a simple aggregate. Both are nullable: a hand-entered record in the
-- "Score a Row" tab has no ground truth, and is excluded from error-rate math.
-- ---------------------------------------------------------------------------
alter table predictions add column if not exists source_row_id bigint references adult_income (id) on delete set null;
alter table predictions add column if not exists true_label    integer;
alter table predictions add column if not exists request_hash  text;

create index if not exists idx_predictions_source_row on predictions (source_row_id);
create index if not exists idx_predictions_true_label on predictions (true_label);

-- ---------------------------------------------------------------------------
-- Row Level Security.
-- 001_init.sql already enabled RLS everywhere and gave anon SELECT on runs.
-- The Bias Audit tab also reads predictions directly with the anon key, so
-- anon needs read access there too. Writes stay server-side with the
-- service-role key, which bypasses RLS entirely.
-- ---------------------------------------------------------------------------
alter table adult_income enable row level security;

drop policy if exists "anon can read adult_income" on adult_income;
create policy "anon can read adult_income"
    on adult_income for select
    to anon
    using (true);

drop policy if exists "anon can read predictions" on predictions;
create policy "anon can read predictions"
    on predictions for select
    to anon
    using (true);