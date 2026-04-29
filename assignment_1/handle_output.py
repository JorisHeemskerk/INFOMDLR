import datetime
import pytz


date = datetime.datetime.now(
    tz=pytz.timezone('Europe/Amsterdam')
).strftime('%d-%m-%Y--%H-%M')
OUTPUT_DIR = f"assignment_4/output/{date}/"
