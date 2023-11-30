from datetime import datetime
def get_insert_dict_index(d, k):
    try:
        v = d[k]
    except:
        v = len(d)
        d[k] = v
    return v

def convert_time(time_string, offset=946659600000):
    try:
        if time_string[-5:].__contains__("."):
            dt_obj = datetime.strptime(time_string,
                                   '%Y.%m.%d.  %H:%M:%S.%f')
        else:
            dt_obj = datetime.strptime(time_string,
                                   '%Y.%m.%d.  %H:%M:%S')

        millisec = int(dt_obj.timestamp() * 1000) - offset
    except:
        millisec = -1
    return millisec
