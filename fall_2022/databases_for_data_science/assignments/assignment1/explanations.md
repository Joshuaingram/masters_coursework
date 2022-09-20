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
This query returns a table of bookings where the dob year is in 1990, 1991, or 1992. Includes all colunmns.

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

This query calculates the average salary for every job_title, and returns name, job_title, employer, salary, and average salary of that job_title for every person through a left outer join. Because of the left outer join, we will have every person listed but they might have null columns if they are not in the employment table.

Steps:
- Find the average salary for every job title in the employment table, call this averages
- Select name column. job_title, employer, and average_salary columns from employment table. average_salary column from averages.
- Left outer join person table with person table, where user_id's are the same
- Left outer join averages with employment, where job_title's are the same

### b. If we changed the left joins to inner joins (i.e., deleted the LEFT keywords), what would be different about the result?

#### Answer:

The query would only return the rows where a user_id has an associated entry in the employment table, and the job_titles for which the average could be calculated.

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

This query finds the oldest person arrested for each charge.

Steps:
- Inner join the charges and booking_dates, where bookingnumbers are the same
- Select charge, name, and dob from this inner join, call it info
- Inner join charges and bookings where bookingnumber the same, group by charge and find the minimum/oldest dob for each charge
- Select the charge and minimum/oldest dob for each charge, call this inner join oldest
- Inner join oldest and info where both the charges are the same and the date of births are the same
- Select the distinct values from dob, charge, and name columns. Order by charge and name and print out only the first 20 rows.

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

Each soid can have multiple addresses. So if we group by address, we still have multiple soid's for the same address... but we don't tell PostgreSQL how to choose soid. PostgreSQL is not told how to select the minimum dob for soid, as it is only being told to group by address. We either need to group by soid, address, or write a subquery.

### b. How does the Problem 3 query avoid this issue?

#### Answer:

We group by the charge in a subquery and select the minimum dob, then we go about inner joining from there.

## Problem 8

### a. How do we know if someone is homeless? This question is exploratory. Show and explain any exploratory queries you use, and cite any external sources of information (e.g. maps, articles). Develop and state a criterion for inferring homelessness.
