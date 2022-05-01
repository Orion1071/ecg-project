import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU'))) 
data = [1,2,3]

model = tf.keras.models.load_model("models/model")
prediction = model.predict(data)
print("0.8414  0.9092  0.1411")
print(prediction)