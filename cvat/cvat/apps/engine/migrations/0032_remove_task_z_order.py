# Generated by Django 3.1.1 on 2020-10-12 17:16

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("engine", "0031_auto_20201011_0220"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="task",
            name="z_order",
        ),
    ]
