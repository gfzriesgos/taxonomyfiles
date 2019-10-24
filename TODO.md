Fix assumptions for schema mapping files.
The current files are only the mappings for the damage states.

So they sum up to 1 for one building in the source taxonomy state,
to those in the other ones (just building type dependent numbers).

Also the very first keys in the conversion matrix are the
target damage states (and not as I thought before the source damage states).

So it is important to change the tests here.

