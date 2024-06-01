from django.db import models

class YourModel(models.Model):
    image = models.ImageField(upload_to='predicted_image/')

    def __str__(self):
        return self.image.name

