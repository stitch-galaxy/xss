from google.cloud import pubsub
from google.cloud import storage

# https://google-auth.readthedocs.io/en/latest/user-guide.html
from google.oauth2 import service_account
credentials = service_account.Credentials.from_service_account_file('stitchgalaxy-78c0704b8fb8.json')

# https://cloud.google.com/storage/docs/access-control/create-signed-urls-program
# from oauth2client.service_account import ServiceAccountCredentials
# credentials = ServiceAccountCredentials.from_json_keyfile_name('stitchgalaxy-78c0704b8fb8.json')

import random
import string
import datetime

import sendgrid
from sendgrid.helpers.mail import *

# project name
PROJECT = 'stitchgalaxy-169408'
# topic and subsciption names
TOPIC = 'job'
SUBSCRIPTION = 'job'
# gcs bucket name
SRC_BUCKET = '{}-job'.format(PROJECT)
DST_BUCKET = '{}-output'.format(PROJECT)

INPUT_FILE_NAME = 'input/input.jpg'
OUTPUT_FILE_NAME = 'output/result.png'

# instantiate clients
pubsub_client = pubsub.Client(PROJECT)
storage_client = storage.Client(PROJECT)

# get subscription
topic = pubsub_client.topic(TOPIC)
subscription = topic.subscription(SUBSCRIPTION)

# get buckets
src_bucket = storage_client.get_bucket(SRC_BUCKET)
dst_bucket = storage_client.get_bucket(DST_BUCKET)

# pull tasks in cycle
while True:
    pulled = subscription.pull(max_messages=1)
    for ack_id, message in pulled:
        try:
            src_blob_path = message.data.decode("utf-8")
            # TODO: remove this line
            src_blob_path = 'input.jpg'
            src_blob = src_bucket.blob(src_blob_path)
            src_blob.download_to_filename(INPUT_FILE_NAME)
            # TODO: NN to process image
            # TODO: get job_id
            job_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
            dst_blob_path = '{}/result.png'.format(job_id)
            dst_blob = dst_bucket.blob(dst_blob_path)
            with open(OUTPUT_FILE_NAME, 'rb') as my_file:
                dst_blob.upload_from_file(my_file)

            # signed url
            signed_url = dst_blob.generate_signed_url(datetime.timedelta(days=1), credentials=credentials)
            print(signed_url)
            # public url
            dst_blob.make_public()
            public_url = dst_blob.public_url
            print(public_url)

            # sending email with send grid
            # TODO: os.environ.get('SENDGRID_API_KEY')
            sg = sendgrid.SendGridAPIClient(apikey='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')
            from_email = Email("test@example.com")
            to_email = Email("tarasov.e.a@gmail.com")
            subject = "Sending with SendGrid is Fun"
            content = Content("text/plain", signed_url)
            mail = Mail(from_email, subject, to_email, content)
            response = sg.client.mail.send.post(request_body=mail.get())
            print(response.status_code)
            print(response.body)
            print(response.headers)

        except:
            print('Error during message processing')
            pass
        subscription.acknowledge([ack_id])





print('Bucket {} created.'.format(src_bucket.name))