from django import forms

class StyleTransferForm(forms.Form):
    content_image = forms.ImageField()
    style_image = forms.ImageField()
