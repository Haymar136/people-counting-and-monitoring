import asyncio
from azure.eventhub.aio import EventHubProducerClient
from azure.eventhub import EventData
import json
import config

connection_string = config.connection_string
event_hub_name = config.event_hub_name

async def send_json_message(connection_string, event_hub_name, json_message):
    # Create an Event Hub producer client
    producer_client = EventHubProducerClient.from_connection_string(
        conn_str=connection_string, eventhub_name=event_hub_name
    )

    async with producer_client:
        # Create a batch.
        event_data_batch = await producer_client.create_batch()

        # Add events to the batch.
        event_data_batch.add(EventData(json.dumps(json_message)))
       
        # Send the batch of events to the event hub.
        await producer_client.send_batch(event_data_batch)

    # Close the producer client
    await producer_client.close()


json_message = {"id": 1, "name": "Bob", "color": "green", "fruit": "apple"}

asyncio.run(send_json_message(connection_string, event_hub_name, json_message))
