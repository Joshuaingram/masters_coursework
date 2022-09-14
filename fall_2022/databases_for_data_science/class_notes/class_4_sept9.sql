/*
Databases for Data Science
Class 4 - Friday, September 9, 2022 at 4PM
Joshua D. Ingram
*/

-- Create a database
-- From bash
-- createdb my_database (creates a database named my_database)
-- Create a database named usr from bash
-- createdb

-- within psql
CREATE DATABASE my_database;

-- Create a table within a psql database
CREATE TABLE student (
    student_id INT,
    name VARCHAR(255),
    bio TEXT,
    on_campus BOOLEAN,
    gpa NUMERIC
);

-- Inserting values
INSERT INTO student VALUES
    (1, 'Alex', '...lives in Sarasota...', FALSE, 3.2),
    (2, 'Blake', 'Aspiring data scientist', FALSE, 3.5);

-- Creating table from query
CREATE TABLE financial_aid AS
    SELECT
        student_id, --column from student
        CAST('2021-08-01' AS DATE) AS start_date, -- convert this string to a date
        (2000 * gpa / 4.0)::MONEY AS scholarship  -- another syntax for casting
    FROM student;

-- Inserting data from a query
INSERT INTO financial_aid
    SELECT
        student_id, -- 
        CURRENT_DATE, --
        500
FROM student
WHERE on_campus;

--
DELETE FROM financial_aid WHERE scholarship IS null;

-- Delete everything from a table
DELETE FROM financial_Aid

-- Delete a table
DROP TABLE financial_aid;

-- Importing data
\copy table_name FROM \location\on\disk WITH CSV HEADER

-- EXERCISE
-- Create a table named orders for the data in
-- cp /usr/share/databases/example/orders.csv
-- Copy that file into your table

-- Create table
CREATE TABLE orders (
    email text,
    name text,
    order_number int,
    order_date date,
    item_price money,
    sku text,
    manufacturer text,
    product_name text,
    variant text,
    gift_wrapped boolean,
    country text,
    state text,
    city text
);

\copy orders FROM /usr/share/databases/example/orders.csv WITH CSV HEADER

-- Update student gpa for Blake
UPDATE student
SET gpa=4.0
WHERE name='Blake';

-- Altering tables (adding last names)
ALTER TABLE student
    RENAME name first_name
    ADD last_name varchar(255);

UPDATE student SET last_name =


-- TRANSACTIONS - A group of operations that either succeeds completely or has no effect

-- Start a transaction =
BEGIN;

-- empty the financial_aid table
DELETE FROM financial_aid;

-- This won't work!
INSERT INTO financial_aid
    VALUES (3, CURRENT_DATE, 1/0);

-- Finish setup and attempt to execute the transaction
COMMIT;

-- ACID Properties
-- Atomicity: A transaction will succeed or fail as a single "operation"; it will not have partial effects
-- Consistency: A transaction must leave the database in a valid state
-- Isolation: Transactions will not interfere with each other.
-- Durability: Once a transaction completes, its effects will not be lost.

-- A database system is ACID Complian if its transactions satisfy these properties.


-- Cancel the transaction
ROLLBACK;

-- Create a savepoint (state)
SAVEPOINT my_save;
-- Cancel everything after the savepoint
ROLLBACK TO my_save;

-- VIEW
-- Views allow you to create a virtual table with the behavior of some query.
-- Every time you reference a view, its underlying query will be run

CREATE VIEW merit_based_aid AS (
    SELECT
        student_id,
        (1000 * gpa \ 4.0) AS scholarship
    FROM
        student
);

-- EXERCISE
-- Create a view that shows the off-campus students and their current total scholarship.
-- 1. Select ???
-- SUM of scholarships from financial_aid and merit_based_aid
-- for students who are off_campus
-- 2. Create view that wraps the select
BEGIN;
CREATE VIEW total_merit_aid AS (
    SELECT student_id,
    SUM(scholayship) AS total
    FROM merit_based_aid
    GROUP BY student_id
);
CREATE VIEW total_aid AS (
    SELECT a.student_id,
    a.total + b.total AS total
    FROM total_merit_aid a
    JOIN total_other_aid b
    ON a.stuent_id = b.student_id
);
COMMIT;