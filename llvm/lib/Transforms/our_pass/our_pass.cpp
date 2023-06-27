#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"



using namespace llvm;

namespace{
    struct OurPass : public ModulePass{
    static char ID;
    OurPass() : ModulePass(ID){}
    bool runOnModule(Module &F) override{
        errs() << "our pass : ";
        
        return false;
    }
}; // end of struct OurPass
} // end of annonymous namespace

char OurPass::ID = 0;
static RegisterPass<OurPass> X("OurPass", "A simple pass for hello world",
                                false,
                                false);