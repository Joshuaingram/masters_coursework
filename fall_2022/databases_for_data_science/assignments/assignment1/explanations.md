# Assignment 1 - SQL Queries

Joshua D. Ingram

Wednesday, September 14, 2022

## Problem 1

```sql

-- homelessness db
SELECT *
FROM bookings
WHERE TO_CHAR(dob, 'YYYY') IN ('1990', '1991', '1992');

```

### a. Write your own explanation of this query.

#### Answer:
This query returns a table containing every column in bookings where the dob year is in 1990, 1991, or 1992.

Steps:
- Find the rows in bookings where the year in dob is in 1990, 1991, 0r in 1992
- Select every column in bookings

## Problem 2

```sql

-- mini_example db
SELECT name,
    employment.job_title,
    employment.employer,
    employment.salary,
    averages.average_salary
FROM employment
    LEFT JOIN person on person.user_id = employment.user_id
    LEFT JOIN (
        SELECT job_title,
            AVG(salary) AS average_salary
        FROM employment
        GROUP BY job_title
    ) averages ON employment.job_title = averages.job_title;


```

### a. Write your own explanation of this query

#### Answer:


Steps:
- 

### b. If we changed the left joins to inner joins (i.e., deleted the LEFT keywords), what would be different about the result?

#### Answer:

## Problem 3

In our second lecture, we attempted to find the *oldest person arrested for each charge*. What follows is a correct solution to this exercise.

```sql
-- homelessness db
SELECT DISTINCT
    info.dob,
    info.charge,
    info.name
FROM ( SELECT
charge,
        MIN(dob) as dob
    FROM
bookings
        JOIN charges ON charges.bookingnumber=bookings.bookingnumber
    GROUP BY charge
) oldest JOIN (
    SELECT
        charge,
        name,
        dob
    FROM
        booking_dates JOIN charges ON booking_dates.bookingnumber=charges.bookingnumber
) info
ON oldest.charge=info.charge AND oldest.dob=info.dob
ORDER BY charge,name
LIMIT 20;
```

### a. Write your own explanation of this query.

- I suggest explaining each of the two subqueries separately and then explainaing how they are joined.

#### Answer:

Steps:
- 

## Problem 4

PostgreSQL will not run the following query.

```sql
-- homelessness db
SELECT
    soid,
    address,
    MIN(dob)
FROM bookings
GROUP BY address;
```

### a. Why doesn't this work? Explain briefly.

#### Answer:

### b. How does the Problem 3 query avoid this issue?

#### Answer:

