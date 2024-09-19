import multiprocessing
import subprocess
import requests
import json
import config
#import counting_people1
#import counting_people2
#import counting_people3

API_ENDPOINT = config.API_ENDPOINT
headers = {'content-type': 'application/json'}

data = {
            'in_count1': config.in_count1,
            'out_count1': config.out_count1,
            'in_time1': config.in_time1,
            'out_time1': config.out_time1,
            'in_count2': config.in_count2,
            'out_count2': config.out_count2,
            'in_time2': config.in_time2,
            'out_time2': config.out_time2,
            'in_count3': config.in_count3,
            'out_count3': config.out_count3,
            'in_time3': config.in_time3,
            'out_time3': config.out_time3,
            'total_in': config.total_in,
            'total_out': config.total_out
        } 
requests.post(url=API_ENDPOINT, data = json.dumps(data), headers=headers)
print(data)

def run_script(script_name):
    subprocess.run(["C:\\Users\\asus\\Documents\\Pythons\\People_counting\\people-counting-main\\myenv\\Scripts\\python", script_name])

if __name__ == "__main__":
    script1 = "counting_people1.py"
    script2 = "counting_people2.py"
    script3 = "counting_people3.py"

    # Create processes for each script
    process1 = multiprocessing.Process(target=run_script, args=(script1,))
    process2 = multiprocessing.Process(target=run_script, args=(script2,))
    process3 = multiprocessing.Process(target=run_script, args=(script3,))

    # Start the processes
    process1.start()
    process2.start()
    process3.start()

    requests.post(url=API_ENDPOINT, data = json.dumps(data), headers=headers)

    # Wait for both processes to finish
    process1.join()
    process2.join()
    process3.join()

    print("Both scripts have finished executing.")
