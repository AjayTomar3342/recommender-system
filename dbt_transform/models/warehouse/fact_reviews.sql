{{ config(materialized='table') }}

SELECT
    -- We create a unique ID by joining the strings with a hyphen
    -- This works perfectly in SQLite without needing external plugins
    user_id || '-' || product_id || '-' || review_timestamp as review_pk,
    user_id,
    product_id,
    rating,
    review_body,
    review_timestamp
FROM {{ ref('stg_amazon_reviews') }}