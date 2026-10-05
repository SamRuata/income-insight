-- 004_audit_function.sql
-- Fairness audit, computed in SQL over predictions JOINed to ground truth.
--
-- The brief asks for false-positive and false-negative rates by protected
-- attribute. Those cannot be read off the predictions table alone: a false
-- positive is only false relative to a true label, which lives on the
-- adult_income row the prediction was made from. Hence the join on
-- source_row_id, and hence the WHERE true_label IS NOT NULL -- a record typed
-- by hand in the UI has no ground truth and must not be counted.
--
-- Defined as a database function (not client-side Python) so the aggregation
-- happens next to the data and the API tier stays thin. Call it with:
--   client.rpc("audit_by_attribute", {"attr": "sex"}).execute()

create or replace function audit_by_attribute(attr text)
returns table (
    group_value     text,
    n               bigint,
    tp              bigint,
    fp              bigint,
    tn              bigint,
    fn              bigint,
    fpr             double precision,
    fnr             double precision,
    positive_rate   double precision,
    accuracy        double precision
)
language plpgsql
security definer
as $$
begin
    -- Only allow grouping by a real protected attribute. Interpolating an
    -- arbitrary string into SQL would be an injection hole; this whitelist
    -- plus %I quoting closes it.
    if attr not in ('sex', 'race', 'workclass', 'marital_status', 'relationship') then
        raise exception 'attribute % is not auditable', attr;
    end if;

    return query execute format($f$
        select
            a.%I::text                                              as group_value,
            count(*)                                                as n,
            count(*) filter (where p.label = 1 and p.true_label = 1) as tp,
            count(*) filter (where p.label = 1 and p.true_label = 0) as fp,
            count(*) filter (where p.label = 0 and p.true_label = 0) as tn,
            count(*) filter (where p.label = 0 and p.true_label = 1) as fn,

            -- FPR: of the people who truly earn <=50K, what share did the
            -- model wrongly flag as >50K?
            (count(*) filter (where p.label = 1 and p.true_label = 0))::float
                / nullif(count(*) filter (where p.true_label = 0), 0)   as fpr,

            -- FNR: of the people who truly earn >50K, what share did the
            -- model miss?
            (count(*) filter (where p.label = 0 and p.true_label = 1))::float
                / nullif(count(*) filter (where p.true_label = 1), 0)   as fnr,

            avg(p.label::float)                                          as positive_rate,
            (count(*) filter (where p.label = p.true_label))::float
                / nullif(count(*), 0)                                    as accuracy
        from predictions p
        join adult_income a on a.id = p.source_row_id
        where p.true_label is not null
        group by a.%I
        order by n desc
    $f$, attr, attr);
end;
$$;

grant execute on function audit_by_attribute(text) to anon, authenticated, service_role;