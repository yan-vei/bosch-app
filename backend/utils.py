from datetime import datetime

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def get_orders_with_time(orders):
    """
    Return orders with their processing times for the last two weeks
    """

    orders_with_times = [[order['datetime_EINGANGSDATUM_UHRZEIT'], order['PROCESSING'] / 86400] for order in orders]

    return orders_with_times


def get_orders_with_priority(orders):
    """
    Return orders with their assigned priority
    """

    priority_count = {}

    for order in orders:
        if order['SONDERFAHRT'] not in priority_count:
            priority_count[order['SONDERFAHRT']] = 1
        else:
            priority_count[order['SONDERFAHRT']] += 1


    return [priority_count]


def get_orders_by_package_type(orders):
    """
    Return orders with their assigned package type
    """
    package_types_df = ['count_PACKSTUECKART=BEH',
                'count_PACKSTUECKART=CAR', 'count_PACKSTUECKART=GBP', 'count_PACKSTUECKART=PAL',
                'count_PACKSTUECKART=PKI', 'count_PACKSTUECKART=UNKNOWN']
    package_types = {"CAR": 0, "BEH": 0, "PAL": 0, "GBP":0}

    for order in orders:
        for type in package_types_df:
            if order[type] == 1:
                if type[-3:] in package_types.keys():
                    package_types[type[-3:]] += 1

    return [package_types]


