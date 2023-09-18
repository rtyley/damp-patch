import os
import time
import ssl
import wifi
import socketpool
import microcontroller
import adafruit_requests
from adafruit_io.adafruit_io import IO_HTTP
from random import randint

#  adafruit quotes URL
quotes_url = "https://www.adafruit.com/api/quotes.php"

#  connect to SSID
wifi.radio.connect(os.getenv('CIRCUITPY_WIFI_SSID'), os.getenv('CIRCUITPY_WIFI_PASSWORD'))

pool = socketpool.SocketPool(wifi.radio)
requests = adafruit_requests.Session(pool, ssl.create_default_context())

while True:
    try:
        #  pings adafruit quotes
        print("Fetching text from %s" % quotes_url)
        #  gets the quote from adafruit quotes
        response = requests.get(quotes_url)
        print("-" * 40)
        #  prints the response to the REPL
        print("Text Response: ", response.text)
        print("-" * 40)
        response.close()
        #  delays for 1 minute
        aio = IO_HTTP(os.getenv('AIO_USERNAME'), os.getenv('AIO_KEY'), requests)

        run_count = 0
        while True:
            print('sending count: ', run_count)
            run_count += 1

            aio.send_data('roberto-test', run_count)
            aio.send_data('rando', randint(0, run_count))
            # Adafruit IO is rate-limited for publishing
            # so we'll need a delay for calls to aio.send_data()
            time.sleep(10)



    # pylint: disable=broad-except
    except Exception as e:
        print("Error:\n", str(e))
        print("Resetting microcontroller in 10 seconds")
        time.sleep(10)
        microcontroller.reset()
