
class ContactForm(forms.Form):

    tweet = forms.CharField(widget=forms.Textarea)

    cc_myself = forms.BooleanField(required=False)