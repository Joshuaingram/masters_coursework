/*
Databases for Data Science
Class 2 - Wednesday, August 31, 2022 5PM
Joshua D. Ingram
*/

/*
Overview

- Review of basic SQL syntax
- Inner and outer joins
- Subqueries
- Data types
- File-based workflows
*/

/*
Commands to connect to NCF cs1 server

- Must be connected to the NCF secure WiFi
- ssh jingram@cs1.ncf.edu
- Enter password
*/

-- new database
CREATEDB <database_name>

-- initialize postgreSQL
psql <database_name>

-- postgreSQL meta-commands
\l -- List databases
\d -- List the tables in the current database
\d my_table -- Describe the coluns in table my_table

-- Basic SQL queries
SELECT * 
FROM person 
WHERE name LIKE "Wiley%"; -- give me every column of the rows in person in which the name column matches the pattern

/*
Nuances

- Postgres keeps reading until it sees a ";".
- SQL is a ~declarative~ language. You are telling the database what you want, not how to find it.
- Each command is interpreted all at once... There must be self consistency, but it is not like linear programming.
- Use the LIMIT function to limit the number of rows we see. (e.g. LIMIT 5)
*/

-- Functions
SELECt to_char(enrollmentdate, "YYYY-MM-DD") as enrolled
FROM students
ORDER BY enrolled;

-- Aggregate Functions
SELECT AVG(income), birth_year
FROM tax_records
GROUP BY birth_year;

-- Joins
-- The JOIN operators combine multiples tables ON shared information (features)

-- Inner Join
SELECT a.bookingnumber, charge, agency
FROM charges a
JOIN bookings b ON a.bookingnumber=b.bookingnumber;

-- RESULT
/*
 bookingnumber |                   charge                   | agency 
---------------+--------------------------------------------+--------
      99066366 | ROBBERY LESS THAN 300                      | HCSO
      99065924 | PETIT THEFT (PRIOR TO 6-8-95)              | TPD
      99062803 | PETIT THEFT (PRIOR TO 6-8-95)              | HCSO
      99062330 | BATTERY ON A LAW ENFORCEMENT OFFICER       | HCSO
      99062006 | POSSESSION OF CANNABIS LESS THAN 20 GRAMS  | HCSO
      99060088 | PROSTITUTION                               | TPD
      99058574 | POSSESSION OF CANNABIS LESS THAN 20 GRAMS  | HCSO
      99057857 | POSSESSION OF COCAINE WITH INTENT TO SELL  | TPD
      99056139 | DRIVING UNDER THE INFLUENCE                | FHP
      99055253 | TRES. ON PROP. OTHER THAN STRUCT. OR CONVE | TPD
      99053605 | UTTERING A FORGED INSTRUMENT               | HCSO
      99048338 | DELIVERY OF CANNABIS WITHIN 1000 FT OF SCH | HCSO
      99048276 | NO VALID DRIVER                            | HCSO
*/

-- The inner joing will combine tables by shared information specified, for every possible matching combination

-- Outer Joins
-- Will comvine tables based on information, regardless of if they are shared
/*
3 Types:
1. Left Outer Join - Joins tables by common elements and all information of left table
2. Right Outer Join - Joins tables by common elements and all information of right table
3. Full Outer Joi - Joins tables with all information
*/

-- Cross Join
SELECT name, city
FROM person
CROSS JOIN residence;
-- cross product of tables... creates every possible combination of values for the given columns

-- Subqueries
SELECT records.name
FROM (
    SELECT *
    FROM person
) AS records;

-- EXAMPLE
-- Using the homelessness database, find the earliest DOB released in each year
SELECT
    to_char(releasedate, 'YYYY') AS year,
    min(dob)
FROM bookings
GROUP BY year
ORDER BY year;

-- RESULTS
/*
 year |    min     
------+------------
 1985 | 1952-10-22
 1986 | 1947-07-29
 1987 | 1953-08-28
 1988 | 1929-01-27
 1989 | 1957-08-13
 1992 | 1966-02-06
 1995 | 1906-02-11
 1996 | 1900-01-01
 1997 | 1901-01-01
 1998 | 1901-01-01
 1999 | 1900-01-01
 2000 | 1901-01-01
 2001 | 1906-01-10
*/

-- EXAMPLE
-- Find the oldest arrestee released in each year
-- We use an inner query
SELECT DISTINCT -- DINSTINCT will get rid of any duplicates
    soid, year, oldest
FROM (
    SELECT
    to_char(releasedate, 'YYYY') AS year,
    min(dob) AS oldest
    FROM bookings
    GROUP BY year
) a
JOIN bookings b ON
    a.oldest=b.dob AND a.year = to_char(b.releasedate, 'YYYY')
ORDER BY year
LIMIT 5;

-- RESULTS
/*
  soid  | year |   oldest   
--------+------+------------
 169182 | 1985 | 1952-10-22
 219341 | 1986 | 1947-07-29
 219284 | 1987 | 1953-08-28
 244599 | 1988 | 1929-01-27
 263749 | 1989 | 1957-08-13
(5 rows)
*/

-- Data Types
/*
Columns in SQL databases are strongly typed. There are many column types in postgreSQL.
For example:
1. boolean
2. int, numeric
3. char(n), varchar(n), text
4. timestamp, date, time, interval

There are many, many more at: 
*/

-- Running SQL from files
/*
You can run SQL from files as follows:

[user@cs1 ~]$ psql -a -d <database_name> -f <filename>

database_name=# \i <filename>

We can also check these different flags/arguments to pass by running "man psql" in bash
*/

-- SIDE NOTE: use "nano <filename>" to open a file with the nano editor... easier than VIM

-- SIDE NOTE: USe the "|" pipe operator with bash to "pipe" different commands

-- Writing results to files
/*
Save the output from executing a file:
[user@cs1 ~]$ psql -a -d <database_name> -f <input_file> -o <output_file>

Set the output of the psql terminal:
database_name=# \o <output_file>

Revert to console output:
database_name=# \o
*/

-- CSV output
/*
[user@cs1 ~]$ psql -a -d <database_name> -f <input_file> -o <output_file> --csv

Or, inside the psql terminal:
database_name=# \copy (SELECT * FROM person) to <output_file> with csv delimiter ',' header

SIDE NOTE: "header" in the code example above includes the column names in the header of the csv file.
*/

-- EXERCISE
-- In homelessness, find the oldest person arrested for each charge
SELECT c.bookingnumber, charge, dob, soid
FROM bookings b
JOIN charges c
    ON c.bookingnumber=b.bookingnumber;

-- MY ATTEMPT
SELECT DISTINCT
dob,
charge,
name
FROM (
    SELECT 
    name,
    bookingnumber, 
    bookings.soid,
    dob
    FROM (
        SELECT arrestees.soid, name
        FROM bookings
        JOIN arrestees
            ON arrestees.soid=bookings.soid
    ) innermost
    JOIN bookings ON
        innermost.soid=bookings.soid
) outermost
JOIN charges
    ON charges.bookingnumber=outermost.bookingnumber
ORDER BY dob
LIMIT 50;

-- ANSWER

SELECT
    dob, charge, name
FROM (
    SELECT oldest.dob, charge, soid FROM (
        -- collect the oldest DoB per charge
        SELECT
            min(dob) as dob,
            charge
        FROM
            bookings
        JOIN charges ON bookings.bookingnumber = charges.bookingnumber
        GROUP BY charge
    ) oldest
    LEFT JOIN bookings ON
        oldest.dob = bookings.dob
) info
JOIN arrestees ON info.soid = arrestees.soid
ORDER BY dob;