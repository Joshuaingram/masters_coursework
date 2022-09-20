/*
# Assignment 1 - SQL Queries

Joshua D. Ingram

Wednesday, September 14, 2022
*/

-- Problem 5
-- Find every row in the booking_dates table where the booking date is on or after January 1, 1990. Sort the results by booking date.

 SELECT * 
 FROM booking_dates
 WHERE bookingdate >= '1991-01-01'
 ORDER BY bookingdate;

 -- Explanation: select rows in booking_dates where bookingdate on or after 1991-01-01, order by bookingdate.

-- Problem 6
-- a. Find every row in the arrestees in which the person's first name is "Mary". 
SELECT *
FROM arrestees
WHERE name LIKE '%,MARY ';

-- Explanation: Select every row where "Mary " comes after the comma (checked without space after Y, but no rows returned).

-- b. List every unique first name in the arrestees table that starts with "Mary".
SELECT *
FROM (
    SELECT
    SPLIT_PART(name,',',2) AS first_name
    FROM arrestees
) names
WHERE names.first_name LIKE 'MARY %';

-- Explanation: Select all distinct first names in arrestees that start with Mary

-- Problem 7
-- a. Using booking_dates, find the age of each person at the date of their first arrest. Display the age in years, months and days.
SELECT 
    ages.soid,
    arrestees.name,
    ages.first_arrest_date,
    ages.age_first_arrest
FROM (
    SELECT
        first_arrests.soid,
        first_arrests.dob,
        first_arrests.first_arrest_date,
        age(first_arrest_date, dob) AS age_first_arrest
    FROM (
        SELECT DISTINCT
            arrests.soid,
            arrests.dob,
            min(arrests.arrestdate) AS first_arrest_date
        FROM (
            SELECT
                bookings.soid,
                bookings.bookingnumber,
                booking_dates.arrestdate,
                booking_dates.dob
            FROM
                bookings
                    JOIN booking_dates ON bookings.bookingnumber = booking_dates.bookingnumber
        ) arrests
        GROUP BY arrests.soid, dob -- I sort of cheat here by assuming the data is clean, and that each SOID always has the same dob if there are multiple entries
    ) first_arrests
) ages
    JOIN arrestees ON ages.soid = arrestees.soid;
/*
EXPLANATION:
- JOIN BOOKINGS AND BOOKING_DATES ON BOOKINGNUMBER, SAVE SOID, BOOKINGNUMBER, AND ARRESTDATE CALL ARRESTS
- GROUP BY SOID, FIND MINIMUM ARRESTDATE FOR EACH SOID, CALL FIRST_ARRESTS
- JOIN FIRST_ARRESTS AND BOOKINGS ON SOID, SAVE FIRST_ARREST_DATE, SOID, DOB, CALL PERSONS_INNER, BOOKINGNUMBER
- TAKE DIFFERENCE IN DOB AND ARRESTDATE ON PERSONS_INNER TO FIND AGE IN YEARS, MONTHS, DAYS USING AGE, CALL DIFFERENCE
- JOIN DIFFERENCE AND ARRESTEES ON SOID, SAVE SOID, NAME, AGE_FIRST_ARREST, BOOKINGNUMBER
*/

-- b. Using charges and booking_dates, find the average age of the people arrested for each charge at the date of arrest.

SELECT
    charge,
    AVG(age_at_arrest) AS charge_average_arrest_age
FROM (
    SELECT 
        soid,
        bookings.bookingnumber,
        bookings.dob,
        arrestdate,
        age(arrestdate, bookings.dob) AS age_at_arrest
    FROM
        booking_dates
            JOIN bookings ON booking_dates.bookingnumber = bookings.bookingnumber
) arrest_ages
    JOIN charges ON charges.bookingnumber = arrest_ages.bookingnumber
GROUP BY charge;


/*
EXPLANATION:
- JOIN BOOKINGS AND BOOKING_DATES ON BOOKINGNUMBER, SAVE SOID, BOOKINGNUMBER, DOB, ARRESTDATE, AND CALCULATE AGE_AT_ARREST CALL ARRESTS
- JOIN ARRESTS AND CHARGES ON BOOKINGNUMBER, GROUP BY CHARGE AND CALCULATE AVERAGE AGE_AT_ARREST
*/

-- c. Find the difference in age between the oldest and youngest age arrested for each charge. Display each difference as a number of years, months and days.
-- This question is not entirely clear. If I am asked for oldest/youngest dob for a given charge, so how old they are now, and not at the date of arrest, I did this right.
-- If we are speaking in terms of oldest/youngest person at day of arrest for any given charge, then I answered this incorrectly..
-- For the latter answer, I would just alter the code from problem 7 b.
SELECT
    charge,
    AGE(MAX(dob), MIN(dob)) AS age_difference
FROM
    bookings
        JOIN charges ON charges.bookingnumber=bookings.bookingnumber
GROUP BY charge

/*
EXPLANATION:
- 

Explanation:
1. List all people arrested for each charge
2. For each person for each charge, calculate their age at arrest date
3. Find the difference in age of the minimum age and the maximum age for each charge, returning difference as number of years, months, and days
*/

-- d. Using booking_dates, COUNT the number of people living at each street address. Ensure that each person is only counted once; use the SOID to identify each person.

SELECT
    count(last_records.soid),
    address
FROM (
    SELECT
        soid,
        max(releasedate) AS last_record
    FROM (
        SELECT
            soid,
            bookings.releasedate,
            bookings.bookingnumber,
            address
        FROM
            booking_dates
                JOIN bookings ON booking_dates.bookingnumber = bookings.bookingnumber
    ) persons
        GROUP BY soid
) last_records
    JOIN bookings ON last_records.last_record = bookings.releasedate AND last_records.soid = bookings.soid
GROUP BY address;


/*
EXPLANATION:
- JOIN BOOKING_DATES AND BOOKINGS ON BOOKINGNUMBER, SAVE SOID, BOOKINGDATE, BOOKINGNUMBER, ADDRESS CALL PERSONS
- GROUP BY SOID, SELECT ROW WHERE BOOKINGDATE MAX, SAVE ADDRESS, SOID
- GROUP BY ADDRESS, COUNT SOIDS
*/


-- Problem 8
-- a. How do we know if someone is homeless? This question is exploratory. Show and explain any exploratory queries you use, and cite any external sources of information (e.g. maps, articles). Develop and state a criterion for inferring homelessness.

SELECT * FROM BOOKINGS WHERE ADDRESS = 'NONE';
SELECT * FROM BOOKINGS WHERE ADDRESS = 'none';
SELECT * FROM BOOKINGS WHERE ADDRESS = 'homeless';
SELECT * FROM BOOKINGS WHERE ADDRESS = 'HOMELESS';
SELECT * FROM BOOKINGS WHERE ADDRESS = 'SHELTER';
SELECT * FROM BOOKINGS WHERE ADDRESS = 'PRISON';
SELECT * FROM BOOKINGS WHERE ADDRESS = 'SALVATION ARMY';
SELECT * FROM BOOKINGS WHERE ADDRESS = 'salvation army';

/*
EXPLANATION:
- LOOK AT ADDRESSES FOR EMPTY VALUES, HOMELESS, COMMAS, NONE, 0s, etc.
*/

-- b. Write a query that returns the columns of bookings with an additional homleessness column (true or false).

SELECT
    *,
    (address NOT IN ('NONE', 'none', 'homeless', 'HOMELESS', 'SHELTER', 'PRISON', 'SALVATION ARMY', 'salvation army')) AS homeless
FROM bookings
ORDER BY homeless;

