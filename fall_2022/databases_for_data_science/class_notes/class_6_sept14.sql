/*
Databases for Data Science
Class 6 - Wednesday, September 14, 2022 at 4PM
Joshua D. Ingram
*/

-- Constraints
/*
Use \d my_table to view the active constraints.
- If you didn't name a constraint, you can see its generated name here.
*/
-- Constraints can be dropped:
ALTER TABLE course_catalog
    DROP CONSTRAINT pandemic,

    -- NOT NULL IS SPECIAL
    ALTER term DROP NOT NULL;

-- If we're using CREATE TABLE my_table AS SELECT ..., how can we assign constraints?
CREATE TABLE financial_aid AS
    SELECT
        student_id,
        (1000.0 * student.gpa)::MONEY
    FROM
        student;

ALTER TABLE my_table
    ADD CONSTRAINT

/*
REVIEW: Normal Forms

1NF

*/

/*
Group Exercise: Normalize your database into 3NF.

Database Name: lec6_adidas

- Design your normalized scheme; draw a rough ER-like diagram
-- Clearly identify primary and foreign keys.
- Write an SQL transaction to create the tables for your design and populate them with data 
-- Add any relevant constraints.
*/

/*
charge_booked
- charge_booked_id INT
- case_id INT
- chargetype_id INT
- carge_id INT

case
- case_id INt
- casenumber TEXT
- booking_id INT
- court_id TEXT

booking
booking_id INT
bookingnumber INT
person_id INT
arrestdate DATE
bookingdate DATE
releasedate DATE
releasecode_id INT
releaseremakrs TEXT

person
person_id INT
soid INT
name TEXT
race TEXT
sex TEXT
dob DATE
e TEXT
*/

BEGIN;

CREATE TABLE CASE (
    case_id SERIAL PRIMARY KEY,
    CASENUMBER TEXT UNIQUE NOT NULL,
    BOOKING_ID INT REFERENCES BOOKING (BOOKING_ID) NOT NULL,
    COURT TEXT NOT NULL
);

CREATE TABLE CHARGE_BOOKED (
    charge_booked_id SERIAL PRIMARY KEY,
    case_id INT UNIQUE NOT NULL,
    chargetype TEXT,
    charge_id INT UNIWUE
);

CREATE TABLE booking (
    booking_id SERIAL PRIMARY KEY,
    bookingnumber INT,
    person_id INT,
    arrestdate DATE,
    bookingdate DATE,
    releasedate DATE,
    releasecode_id INT,
    releaseremarks TEXT
);

CREATE TABLE person (
    person_id SERIAL PRIMARY KEY,
    soid INT,
    name TEXT,
    sex TEXT,
    dob DATE,
    e TEXT
);

COMMIT;