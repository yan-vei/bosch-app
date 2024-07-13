import os
from fastapi import FastAPI, Request
import motor.motor_asyncio
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from datetime import datetime, timedelta
from utils import get_orders_with_time, get_orders_with_priority, get_orders_by_package_type
from model.predict import make_prediction

load_dotenv()


app = FastAPI(
    title="Bosch Order Processing Backend"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = motor.motor_asyncio.AsyncIOMotorClient(os.environ['MONGODB_URL'])
db = client.get_database('test')
bosch_data = db.get_collection('bosch-dataset')


@app.get(
    "/orders",
    response_description="List all order statistics"
)
async def get_orders_stats():
    """
        Get all orders and output relevant statistics:
        - average processing time for the last two weeks
        - total average processing time
    """

    # Fetch all orders from the database
    orders = await bosch_data.find().to_list(length=None)

    # Keep only the orders within the last 2 weeks
    date_format = "%Y-%m-%d %H:%M:%S"

    # To work in production, uncomment this. However, we don't have data from the last two weeks.
    # now = datetime.now()

    now = datetime.now() - timedelta(days=120)
    two_weeks_ago = now - timedelta(weeks=2)
    filtered_orders = [
        order for order in orders
        if datetime.strptime(order['datetime_EINGANGSDATUM_UHRZEIT'], date_format) >= two_weeks_ago and
           datetime.strptime(order['datetime_EINGANGSDATUM_UHRZEIT'], date_format) <= now
    ]
    # Get average processing time by summing all processing times, dividing by the amount of orders and finally converting
    # to days
    average_processing_time = (sum(order["PROCESSING"] for order in filtered_orders) / len(filtered_orders)) / 86400
    total_average_processing_time = (sum(order["PROCESSING"] for order in orders) / len(orders)) / 86400

    # Get last two weeks orders + processing times
    orders_with_times = get_orders_with_time(filtered_orders)
    orders_with_priorities = get_orders_with_priority(filtered_orders)
    orders_by_package_type = get_orders_by_package_type(filtered_orders)

    return {'avg_process_time': average_processing_time, 'total_avg_processing_time': total_average_processing_time,
            'orders': orders_with_times, 'priorities': orders_with_priorities, "orders_by_package_type": orders_by_package_type}


@app.post(
    "/orders",
    response_description="Make a prediction for the order"
)
async def simulate_prediction(request: Request):
    data = await request.json()

    # Generate order creation time data
    current_timestamp = datetime.now()
    formatted_date = current_timestamp.strftime('%Y%m%d')
    time_of_day_seconds = current_timestamp.hour * 3600 + current_timestamp.minute * 60 + current_timestamp.second
    day_of_week = current_timestamp.weekday()

    user_input = [int(formatted_date), int(time_of_day_seconds), int(day_of_week), int(data['priority'])]
    processing_time = make_prediction(user_input)

    return {'processing_time': processing_time}
