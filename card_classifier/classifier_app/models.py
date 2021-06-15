from django.db import models


class Session(models.Model):
    start_time = models.DateTimeField('Start time')
    OCR_results = models.JSONField(default=dict)
    reference_frame = models.ImageField()
    config_file = models.JSONField(default=dict, null= True)


class circleHologram(models.Model):
    session = models.ForeignKey(Session, on_delete=models.CASCADE)
    # circle_hologram_detected = models.BooleanField('Circle Hologram Detected')
    frame_number = models.IntegerField(default=0)
    detection_number = models.IntegerField(null=True)


class flagHologram(models.Model):
    session = models.ForeignKey(Session, on_delete=models.CASCADE, null=False)
    flag_image = models.BinaryField(null=True)
    session_variance = models.FloatField(default=0.0, verbose_name="Variance after current frame")
    distance = models.FloatField(default=0.0)


class extractedData(models.Model):
    session = models.ForeignKey(Session, on_delete=models.CASCADE, null=False)
    data = models.JSONField()
