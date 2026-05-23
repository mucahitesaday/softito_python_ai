-- create
CREATE TABLE kullanicilar (
    id SERIAL PRIMARY KEY,
    ad VARCHAR(50),
    email VARCHAR(100)
);

-- insert
INSERT INTO kullanicilar (ad,email)
VALUES ('Begum', 'tambirzkusagi@gmail.com');

INSERT INTO kullanicilar (ad,email)
VALUES ('Begum', 'sizbirdebenitwitterdagorun@gmail.com');

-- fetch
SELECT * FROM kullanicilar;

SELECT ad, email
FROM kullanicilar
WHERE id=1;

-- update
UPDATE kullanicilar
SET ad = 'İsmail'
WHERE id=1;

SELECT * FROM kullanicilar;

-- delete
DELETE
FROM kullanicilar
WHERE id=1;

SELECT * FROM kullanicilar;