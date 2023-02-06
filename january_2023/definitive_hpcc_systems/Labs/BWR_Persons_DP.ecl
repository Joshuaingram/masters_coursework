IMPORT $;
IMPORT STD;
Persons := $.File_Persons.File;

persons_profile := STD.DataPatterns.Profile(Persons);

OUTPUT(persons_profile)
