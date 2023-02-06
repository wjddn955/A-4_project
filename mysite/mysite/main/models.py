from django.db import models

# Create your models here.

class Diary_content(models.Model):
    para = models.CharField(max_length = 1000, blank=False)