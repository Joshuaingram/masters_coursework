-- Which charges have the most repeat offenders?
-- 1. Which person was arrested for each record in `charges`?
with soid_charges as (
    select bookings.soid,
        charges.charge
    from bookings
        join charges on bookings.bookingnumber = charges.bookingnumber
),

-- 2. How many times was each person charged with each offense?
charge_counts as (
    select soid,
        charge,
        count(*) as num
    from soid_charges
    group by soid,
        charge
)

-- 3. Per charge, how many people were charged more than once?
select charge,
    count(*) as num
from charge_counts
where charge_counts.num >= 2
group by charge -- sort from most to least repeat offenders
order by num desc;