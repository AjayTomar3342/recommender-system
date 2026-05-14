SELECT
    json_extract(RAW_CONTENT, '$.reviewerID') as user_id,
    json_extract(RAW_CONTENT, '$.asin') as product_id,
    CAST(json_extract(RAW_CONTENT, '$.overall') AS FLOAT) as rating,
    json_extract(RAW_CONTENT, '$.reviewText') as review_body,
    datetime(json_extract(RAW_CONTENT, '$.unixReviewTime'), 'unixepoch') as review_timestamp
FROM {{ source('main', 'RAW_REVIEWS') }}