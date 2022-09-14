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
ORDER BY dob;