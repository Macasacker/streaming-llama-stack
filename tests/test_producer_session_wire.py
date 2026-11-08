"""Compile the producer's private session transport, without a model.

The producer keeps the payload pipe in a duplicated descriptor and turns its own
stdout into stderr, so its diagnostics cannot land in the session's JSON stream.
That is two syscalls and one return-value convention: POSIX dup2 reports the
descriptor it installed, where the deleted Win32 _dup2 reported 0. Reading the
production class and running it over real pipes is the only way to tell the two
apart.
"""
import os
from pathlib import Path
import shutil
import subprocess
import unittest
import uuid

TOOL = Path('engines/streaming-llama/prefill_llama.cpp/tools/dsv41-prefill')

PROGRAM = r'''
#include <cstdio>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <unistd.h>

#include "producer-wire.h"

static void require(bool ok,const char*why){if(!ok)throw std::runtime_error(why);}

__CLASS__

int main(int argc,char**argv){
    require(argc==2,"usage: fixture <stderr log>");
    int in_pipe[2],out_pipe[2];
    if(pipe(in_pipe)||pipe(out_pipe))return 10;
    const int log=open(argv[1],O_WRONLY|O_CREAT|O_TRUNC,0600);
    if(log<0)return 11;
    // The shape the parent hands the producer: private pipes on stdio, log on
    // stderr, so the payload has one end and the diagnostics another.
    if(dup2(in_pipe[0],STDIN_FILENO)<0||dup2(out_pipe[1],STDOUT_FILENO)<0||dup2(log,STDERR_FILENO)<0)return 12;
    close(in_pipe[1]);
    try{
        session_output_redirect redirect;
        redirect.bind();
        const std::string payload="PAYLOAD";
        require(producer_wire::is_pipe(redirect.get()),"retained payload end is not a pipe");
        require(producer_wire::write_all(redirect.get(),payload.data(),payload.size()),"payload write failed");
        std::printf("DIAG-VIA-STDOUT\n");
        std::fflush(stdout);
        unsigned char seen[payload.size()];
        size_t got=0;
        while(got<sizeof(seen)){
            const ssize_t read=::read(out_pipe[0],seen+got,sizeof(seen)-got);
            if(read<=0)return 13;
            got+=size_t(read);
        }
        // Diagnostic stdout must not have reached the payload, and the payload
        // must not have been swallowed by the redirection.
        if(std::string(reinterpret_cast<char*>(seen),sizeof(seen))!=payload)return 14;
        fprintf(stderr,"fixture: payload isolated from diagnostic stdout\n");
        fflush(stderr);
    }catch(const std::exception&e){
        fprintf(stderr,"bind or isolation failed: %s\n",e.what());
        return 1;
    }
    return 0;
}
'''


PIPE_PROGRAM = r'''
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>

#include "producer-wire.h"

static void require(bool ok,const char*why){if(!ok)throw std::runtime_error(why);}
// The production session asks a cancellation flag between reads; the fixture has none.
static void stop(){}
// The class also offers a JSON reply helper. This fixture exercises the read side only,
// so a stub keeps the production class compiling without pulling in nlohmann.
struct json{std::string dump()const{return std::string();}};

__CLASS__

int main(int argc,char**argv){
    require(argc==2,"usage: fixture <result log>");
    int in_pipe[2],out_pipe[2];
    if(pipe(in_pipe)||pipe(out_pipe))return 10;
    // The parent's shape: private pipes inherited as stdio.
    if(dup2(in_pipe[0],STDIN_FILENO)<0||dup2(out_pipe[1],STDOUT_FILENO)<0)return 12;
    close(in_pipe[0]);
    FILE*log=fopen(argv[1],"w");
    if(!log)return 11;
    try{
        session_pipe transport(5000,producer_wire::standard_output());
        // The parent is silent far longer than the transport's 10 ms poll cadence.
        // Silence is the parent composing its next command, not a closed pipe: the
        // wait must renew and then read the line that arrives.
        std::thread parent([&]{
            std::this_thread::sleep_for(std::chrono::milliseconds(400));
            const std::string line="ping\n";
            if(!producer_wire::write_all(in_pipe[1],line.data(),line.size()))_exit(20);
        });
        const std::string got=transport.line();
        parent.join();
        require(got=="ping","idle-parent poll timeout must not close the session");
        fprintf(log,"idle parent survived the poll cadence\n");
        // A parent that really goes away still reports itself, within the deadline.
        close(in_pipe[1]);
        try{
            const std::string second=transport.line();
            fprintf(log,"closed pipe went unseen: %s\n",second.c_str());
            fclose(log);return 21;
        }catch(const std::exception&e){
            const std::string why=e.what();
            if(why!="Session parent input pipe closed"){
                fprintf(log,"closed pipe reported %s\n",why.c_str());fclose(log);return 22;
            }
            fprintf(log,"closed parent reported the closure\n");
        }
        fflush(log);fclose(log);
    }catch(const std::exception&e){
        fprintf(log,"session pipe failed: %s\n",e.what());fflush(log);fclose(log);return 1;
    }
    return 0;
}
'''


def production_class(source, marker='class session_output_redirect'):
    found = source.index(marker)
    start = source.index('{', found)
    depth = 0
    for index in range(start, len(source)):
        depth += (source[index] == '{') - (source[index] == '}')
        if depth == 0:
            return source[found:index + 1] + ';'
    raise AssertionError('Unclosed session_output_redirect')


class ProducerSessionWireTests(unittest.TestCase):
    def test_bind_isolates_diagnostic_stdout_from_the_payload_pipe(self):
        if os.name == 'nt':
            self.skipTest('The producer wire is the POSIX twin of the Win32 layer')
        compiler = os.environ.get('SLSTACK_CXX') or shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
        if not compiler:
            self.skipTest('A C++ compiler is required for the session transport regression')
        root = Path(__file__).resolve().parents[1]
        source = (root / TOOL / 'live-session.inc').read_text(encoding='utf-8')
        temporary = Path(os.environ.get('SLSTACK_TEST_ROOT', root / 'runtime')) / ('producer-wire-' + uuid.uuid4().hex)
        temporary.mkdir(parents=True)
        fixture = temporary / 'wire.cpp'
        log = temporary / 'stderr.log'
        executable = temporary / 'wire'
        fixture.write_text(PROGRAM.replace('__CLASS__', production_class(source)), encoding='utf-8')
        try:
            compiled = subprocess.run([compiler, '-std=c++17', '-O1', '-I', str(root / TOOL),
                                      str(fixture), '-o', str(executable)],
                                      capture_output=True, text=True, timeout=120)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            result = subprocess.run([str(executable), str(log)], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            written = log.read_text(encoding='utf-8')
            self.assertIn('DIAG-VIA-STDOUT', written)      # stdout is stderr now
            self.assertNotIn('PAYLOAD', written)           # and the payload never joined it
        finally:
            for file in temporary.iterdir():
                file.unlink()
            temporary.rmdir()


    def test_poll_timeout_renews_the_wait_and_a_closed_parent_still_reports(self):
        """A silent parent is not a closed pipe.

        The transport polls its input every 10 ms to keep the deadline and the
        cancellation flag live between reads. Treating a poll timeout as closure
        ended every persistent session as soon as the parent took longer than
        10 ms to send the next command, which is what made a completed prefill
        answer 'Session parent input pipe closed'.
        """
        if os.name == 'nt':
            self.skipTest('The producer wire is the POSIX twin of the Win32 layer')
        compiler = os.environ.get('SLSTACK_CXX') or shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
        if not compiler:
            self.skipTest('A C++ compiler is required for the session transport regression')
        root = Path(__file__).resolve().parents[1]
        source = (root / TOOL / 'live-session.inc').read_text(encoding='utf-8')
        temporary = Path(os.environ.get('SLSTACK_TEST_ROOT', root / 'runtime')) / ('producer-pipe-' + uuid.uuid4().hex)
        temporary.mkdir(parents=True)
        fixture = temporary / 'pipe.cpp'
        log = temporary / 'result.log'
        executable = temporary / 'pipe'
        fixture.write_text(PIPE_PROGRAM.replace('__CLASS__', production_class(source, 'class session_pipe')), encoding='utf-8')
        try:
            compiled = subprocess.run([compiler, '-std=c++17', '-O1', '-I', str(root / TOOL),
                                      str(fixture), '-o', str(executable), '-pthread'],
                                      capture_output=True, text=True, timeout=180)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            result = subprocess.run([str(executable), str(log)], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            written = log.read_text(encoding='utf-8')
            self.assertIn('idle parent survived the poll cadence', written)
            self.assertIn('closed parent reported the closure', written)
        finally:
            for file in temporary.iterdir():
                file.unlink()
            temporary.rmdir()


if __name__ == '__main__':
    unittest.main()
