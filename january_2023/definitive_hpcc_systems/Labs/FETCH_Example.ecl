IMPORT $;
Person := $.File_People;
OUTPUT(Person,NAMED('Original_File'));
PtblRec := RECORD
  STRING2  State  := Person.state;
  STRING20 City   := Person.city;
  STRING25 Lname  := Person.lastname;
  STRING15 Fname  := Person.firstname;
	STRING1  Gender := Person.Gender;
END;

DataFile := '~BFTEMP::TestFetch';
KeyFile  := '~BFTEMP::TestFetchKey';

PtblOut := OUTPUT(TABLE(Person(Lastname[1..3]='Wik'),PtblRec),,DataFile,OVERWRITE,NAMED('WikTable'));

Ptbl_Plus := DATASET(DataFile,
		                 {PtblRec,UNSIGNED8 RecPos {VIRTUAL(fileposition)}},
		                 THOR);

AlphaInStateCity := INDEX(Ptbl_Plus,
				                  {state,city,lname,fname,RecPos},
				                  KeyFile);

Bld := BUILD(AlphaInStateCity,OVERWRITE);


// AlphaPeople := FETCH(Ptbl_Plus, 
    										 // AlphaInStateCity(WILD(state),
    																			// WILD(city),
    																			// KEYED(Lname='Wikoff')
   																			// ), 
             					 // RIGHT.RecPos);

 

										 
AlphaPeople := FETCH(Ptbl_Plus, 
             					 AlphaInStateCity(Lname='Wikoff'), 
            						 RIGHT.RecPos);										 





OutFile := OUTPUT(CHOOSEN(AlphaPeople,10),NAMED('FETCH_Output10'));
Test := AlphaInStateCity(State = 'MI');

SEQUENTIAL(PtblOut,Bld,OutFile);
Test;
