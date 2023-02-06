 MyRec := RECORD
	STRING1 Value1;
	STRING1 Value2;
END;

SomeFile := DATASET([{'C','G'},
                     {'C','C'},
                     {'A','X'},
                     {'B','G'},
                     {'A','B'}],MyRec);

MyOutRec := RECORD
	myRec.Value1;    //no default value
	SomeFile.Value2; //default value - PROJECT does not care!
	STRING4 CatValues;
  //Record below for TABLE in Exercise 4B
	// UNSIGNED4 RecID := 0; 
	// $.File_Accounts.File;
END;

MyOutRec CatThem(SomeFile Le, INTEGER Cnt) := TRANSFORM
	SELF.CatValues := Le.Value1 + Le.Value2 + '-' + Cnt;
	// SELF := Le;									
 SELF.value1 := Le.value1;
 SELF.value2 := Le.value2;
END;

CatRecs := PROJECT(SomeFile,CatThem(LEFT,COUNTER));
OUTPUT(CatRecs);

/* CatRecs result set is:
	Rec#	Value1	Value2	CatValues
	1		C		G		CG-1
	2		C		C		CC-2
	3		A		X		AX-3
	4		B		G		BG-4
	5		A		B		AB-5
*/