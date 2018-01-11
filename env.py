from google.cloud import bigquery

client = bigquery.Client()


QUERY_W_PARAM = (
'SELECT i.image_id AS image_id, original_url, confidence'
'FROM `bigquery-public-data.open_images.labels` l'
'INNER JOIN `bigquery-public-data.open_images.images` i'
'ON l.image_id = i.image_id'
'WHERE label_name=@label AND confidence >= 0.85 AND Subset=@subset'
'LIMIT 10')

TIMEOUT = 30  # in seconds
param1 = bigquery.ScalarQueryParameter('label', 'STRING', '/m/0f8sw')
param2 = bigquery.ScalarQueryParameter('subset', 'STRING', 'train')
job_config = bigquery.QueryJobConfig()
job_config.query_parameters = [param1, param2]
query_job = client.query(
    QUERY_W_PARAM, job_config=job_config)  # API request - starts the query

# Waits for the query to finish
iterator = query_job.result(timeout=TIMEOUT)
rows = list(iterator)

assert query_job.state == 'DONE'
assert len(rows) == 100
row = rows[0]
assert row[0] == row.name == row['name']
assert row.state == 'TX'
