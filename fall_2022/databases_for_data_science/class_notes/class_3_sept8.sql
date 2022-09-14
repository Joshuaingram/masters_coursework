/*
Databases for Data Science
Class 3 - Thursday, September 8, 2022 4PM
Joshua D. Ingram
*/

-- Matrching Multiple Values
WHERE ... IN (...) -- matrch any of a list of values

/*
Asking Useful Questions

Questions about the dataset
- Why are there more charges than bookings?
- Why are there fewer booking_dates than bookings?
- What do the column names mean?
- How reliable is the dataset? Do certain counties not report this information?

Questions of methodology
- What signifies homelessness? Lack if address?
- Can homelessness status be given by the data?

Questions answerable in the dataset
- Who has been arressted over ten times?
- How many arrests are there by race, gender, or ethnicity?
- What is the number of charges per county per year?
- What is the frequency of charges by month?
*/

-- 1. What are the least arrested charges?

SELECT charge,
    count(*) AS num
FROM charges
GROUP BY charge
ORDER BY num;


-- 2. Which charges have the most offenders?

-- Which person was arrested for each record in offenders?

WITH SOID_CHARGES AS (

)

-- How many times was each person charged with each offense?

-- Per charge, how many people were charged more than once?

-- 3. For each change, what percentage of arrests were in each demographic category?