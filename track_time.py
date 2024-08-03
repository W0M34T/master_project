# Track CPU/RAM
import threading
import psutil
import time
from datetime import datetime

def display_cpu():
    global running
    global cpu_total
    global ram_total
    global start_time

    running = True
    cpu_total = 0
    ram_total = 0
    start_time = time.strftime('%H%M%S')

    # start loop
    while running:
        "CPU-Auslastung oder -Nutzung bezeichnet die Zeit, die ein Computer benötigt, um bestimmte Informationen zu verarbeiten."
        cpu_temp = psutil.cpu_percent(interval=1)
        # print('Die CPU-Auslastung beträgt:', cpu_temp)
        cpu_total += cpu_temp

        "RAM-Auslastung oder HAUPTSPEICHER-AUSLASTUNG bezeichnet dagegen die Zeit, die RAM von einem bestimmten System zu einem bestimmten Zeitpunkt genutzt wird."
        ram_temp = psutil.virtual_memory()[2]
        # print('RAM memory % used:', ram_temp)
        ram_total += ram_temp

def start():
    global t

    # create thread and start it
    t = threading.Thread(target=display_cpu)
    t.start()
    
def stop():
    global running
    global end_time
    global t

    end_time = time.strftime('%H%M%S')

    # use `running` to stop loop in thread so thread will end
    running = False

    # wait for thread's end
    t.join()

    print("############################################")
    print("############################################")
    print("############################################")
    print("############################################")
    cpu_temp = round(cpu_total, 2)
    ram_temp = round(ram_total, 2)
    print(f"cpu_total: {cpu_temp}")
    print(f"ram_total: {ram_temp}")

    t1 = datetime.strptime(start_time, '%H%M%S')
    t2 = datetime.strptime(end_time, '%H%M%S')
    duration = t2 - t1
    seconds = duration.total_seconds()

    if seconds > 0:
        cpu_usage = round(cpu_total / seconds, 2)
        ram_usage = round(ram_total / seconds, 2)
        ram = round(ram_usage / 100 * 32, 2)

        print(f"cpu: {cpu_usage}\nram: {ram_usage} % {ram} GB\ntime: {seconds} seconds")