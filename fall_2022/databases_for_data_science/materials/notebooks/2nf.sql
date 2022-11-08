```sql
CREATE TABLE user (
    user_id SERIAL PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL
);

CREATE TABLE order (
    order_number SERIAL PRIMARY KEY,
    user_id INT NOT NULL REFERENCES user(user_id),
    order_date DATE NOT NULL,
    country TEXT NOT NULL,
    state TEXT,
    city TEXT NOT NULL
);

-- hmm
CREATE TABLE item (
    item_id SERIAL PRIMARY KEY,
    manufacturer TEXT NOT NULL,
    product_name TEXT NOT NULL,
    sku TEXT NOT NULL,
    variant TEXT,
    item_price MONEY NOT NULL
);

CREATE TABLE order_item (x
    order_item_id SERIAL PRIMARY KEY,
    order_number INT NOT NULL REFERENCES order(order_number),
    item_id INT NOT NULL REFERENCES item(item_id),
    gift_wrapped BOOLEAN NOT NULL DEFAULT FALSE
);
```